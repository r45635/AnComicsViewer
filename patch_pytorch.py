import torch.serialization

# Sauvegarde de la fonction originale
_original_load = torch.serialization.torch.load

def patched_load(*args, **kwargs):
    """Fonction patch pour torch.load avec weights_only=False par défaut"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# Application du patch
torch.serialization.torch.load = patched_load
torch.load = patched_load
