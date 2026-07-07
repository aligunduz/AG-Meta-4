from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d


def make(enc_name, enc_args, clf_name, clf_args):
  """
  Initializes a random meta model.

  Args:
    enc_name (str): name of the encoder (e.g., 'resnet12').
    enc_args (dict): arguments for the encoder.
    clf_name (str): name of the classifier (e.g., 'meta-nn').
    clf_args (dict): arguments for the classifier.

  Returns:
    model (MAML): a meta classifier with a random encoder.
  """
  enc = encoders.make(enc_name, **enc_args)
  clf_args['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(clf_name, **clf_args)
  model = MAML(enc, clf)
  return model


def load(ckpt, load_clf=False, clf_name=None, clf_args=None):
  """
  Initializes a meta model with a pre-trained encoder.

  Args:
    ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
    load_clf (bool, optional): if True, loads a pre-trained classifier.
      Default: False (in which case the classifier is randomly initialized)
    clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
    clf_args (dict, optional): arguments for the classifier.
    (The last two arguments are ignored if load_clf=True.)

  Returns:
    model (MAML): a meta model with a pre-trained encoder.
  """
  enc = encoders.load(ckpt)
  if load_clf:
    clf = classifiers.load(ckpt)
  else:
    if clf_name is None and clf_args is None:
      clf = classifiers.make(ckpt['classifier'], **ckpt['classifier_args'])
    else:
      clf_args['in_dim'] = enc.get_out_dim()
      clf = classifiers.make(clf_name, **clf_args)
  model = MAML(enc, clf)
  if 'gradient_transport_state_dict' in ckpt:
    model.gradient_transport_logits.load_state_dict(
      ckpt['gradient_transport_state_dict'])
  return model


class MAML(Module):
  def __init__(self, encoder, classifier):
    super(MAML, self).__init__()
    self.encoder = encoder
    self.classifier = classifier
    self.gradient_transport_logits = nn.ParameterDict()
    self._gradient_transport_names = {}
    device = next(self.parameters()).device

    for name, _ in self.encoder.named_parameters():
      key = 'encoder__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(
        torch.tensor(4.0, device=device))
      self._gradient_transport_names[key] = 'encoder.' + name

    for name, _ in self.classifier.named_parameters():
      # 'temp' is never part of the inner-loop fast weights (see forward()),
      # so a gate for it would never be applied to any gradient.
      if name == 'temp':
        continue
      key = 'classifier__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(
        torch.tensor(4.0, device=device))
      self._gradient_transport_names[key] = 'classifier.' + name

  def reset_classifier(self):
    self.classifier.reset_parameters()

  def get_gradient_transport_gates(self, frozen=()):
    """
    Args:
      frozen (list, optional): parameter-name substrings excluded from the
        inner loop (see inner_args['frozen'] in forward()). Gates for those
        parameters are never applied to a gradient, so they are omitted here
        to avoid diluting logged/averaged gate statistics.
    """
    out = {}
    for key, logit in self.gradient_transport_logits.items():
      name = self._gradient_transport_names.get(key, key)
      if any(s in name for s in frozen):
        continue
      out[key] = torch.sigmoid(logit).detach().item()
    return out

  def _inner_forward(self, x, params, episode):
    feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits

  def _inner_iter(
          self,
          x,
          y,
          params,
          mom_buffer,
          episode,
          inner_args,
          detach,
          use_gradient_transport=False):
    """
    Performs one inner-loop iteration of MAML, optionally applying a learned
    scalar gradient transport gate per parameter tensor.
    """
    with torch.enable_grad():
      logits = self._inner_forward(x, params, episode)
      loss = F.cross_entropy(logits, y)
      grads = autograd.grad(loss, params.values(),
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)

      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          if use_gradient_transport:
            gate_key = name.replace('.', '__')
            gate = torch.sigmoid(self.gradient_transport_logits[gate_key])
            grad = gate * grad
          updated_param = param - lr * grad
        if detach:
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _adapt(
          self,
          x,
          y,
          params,
          episode,
          inner_args,
          meta_train,
          use_gradient_transport=False):
    """
    Performs inner-loop adaptation in MAML.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)

    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    def _inner_iter_cp(episode, *state):
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach,
        use_gradient_transport=use_gradient_transport)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._inner_iter(
          x, y, params, mom_buffer, episode, inner_args, not meta_train,
          use_gradient_transport=use_gradient_transport)

    return params

  def forward(
          self,
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train,
          use_gradient_transport=False):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.

    Returns:
      logits (float tensor, [n_episode, n_way * n_query, n_way]): query logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + [
          'temp', 'gradient_transport_logits']):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)):
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()

      updated_params = self._adapt(
        x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train,
        use_gradient_transport=use_gradient_transport)

      with torch.set_grad_enabled(meta_train):
        self.eval()
        logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
      logits.append(logits_ep)

    self.train(meta_train)
    return torch.stack(logits)
