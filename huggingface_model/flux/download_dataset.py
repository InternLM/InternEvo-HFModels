from datasets import load_dataset, load_from_disk
import torchvision.transforms as transforms

transform = transforms.ToTensor()
dataset = load_dataset('kadirnar/fluxdev_controlnet_16k')
breakpoint()
# dataset.save_to_disk('/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/flux-dev')
# print("here")

# dataset = load_from_disk('/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/train')

length = len(dataset['json_data'])

for i in range(length):
    img = dataset['image'][i]
    tensor_img = transform(img)
    breakpoint()
    text = dataset['json_data'][i]

print("here")

