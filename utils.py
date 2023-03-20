import torch
import numpy as np
import matplotlib.pyplot as plt

def log_images(batch, generator, discriminator, tensor_type):

	ct = batch["ct"]
	input = batch["img"].type(tensor_type)
	structure_mask = batch["structure_masks"].sum(dim=1)
	possible_dose_mask = batch["possible_dose_mask"]
	real_dose = batch["dose"]

	with torch.no_grad():
		generator.eval()
		fake_dose = generator(input)
		if discriminator is not None:
			discriminator.eval()
			proba = torch.nn.Sigmoid()(discriminator(input, fake_dose))[:, 0]
		else:
			proba = torch.zeros(size=(fake_dose.size(0), 1))

	fig = plt.figure(figsize=(30, 30))
	for i in range(fake_dose.size(0)):
		plt.subplot(fake_dose.size(0), 5, 5 * i + 1)
		plt.imshow(ct[i], cmap='gray', origin='lower', vmin=0)
		plt.title("CT")
		plt.axis('off')

		plt.subplot(fake_dose.size(0), 5, 5 * i + 2)
		plt.imshow(possible_dose_mask[i], cmap='gray', origin='lower', vmin=0)
		plt.title("Possible dose mask")
		plt.axis('off')

		plt.subplot(fake_dose.size(0), 5, 5 * i + 3)
		plt.imshow(structure_mask[i], cmap='gray', origin='lower', vmin=0)
		plt.title("Structure masks")
		plt.axis('off')

		plt.subplot(fake_dose.size(0), 5, 5 * i + 4)
		plt.imshow(real_dose[i], cmap='gray', origin='lower', vmin=0)
		plt.title("Dose")
		plt.axis('off')

		plt.subplot(fake_dose.size(0), 5, 5 * i + 5)
		plt.imshow(fake_dose[i, 0, :, :].cpu().detach().numpy(), cmap='gray', origin='lower', vmin=0, vmax=255)
		plt.title(f"Predicted dose (proba: {proba[i].item():.2f})")
		plt.axis('off')

	return fig