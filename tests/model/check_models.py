import torch

checkpoint1 = torch.load('../models/character_classifier_optimized_best.pth', map_location='cpu')
checkpoint2 = torch.load('../models/character_classifier_best.pth', map_location='cpu')

print('Model 1 (optimized) classifier keys:')
for key in sorted(checkpoint1['model_state_dict'].keys()):
    if 'classifier' in key:
        print(f'  {key}: {checkpoint1["model_state_dict"][key].shape}')

print('\nModel 2 (best) classifier keys:')
for key in sorted(checkpoint2['model_state_dict'].keys()):
    if 'classifier' in key:
        print(f'  {key}: {checkpoint2["model_state_dict"][key].shape}')