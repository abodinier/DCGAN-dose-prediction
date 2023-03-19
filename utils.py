import matplotlib.pyplot as plt

def save_image(imgs, gen_img, n_epoch, path_save_fig = f"./images/generator/epoch"):

  real_ct = imgs["ct"] #.type(Tensor)
  real_structure_masks = imgs["structure_masks"].sum(axis=1) #.type(Tensor)
  real_dose = imgs["dose"] #.type(Tensor)
  real_img = imgs["img"] #.type(Tensor) 
  real_possible_dose_mask = imgs["possible_dose_mask"] #.type(Tensor)

  n_batchs = real_ct.shape[0]

  for i_batch in range(n_batchs):

    real_possible_dose_mask_data_i = real_possible_dose_mask.data[i_batch,:,:]
    real_ct_data_i = real_ct.data[i_batch,:,:]
    real_dose_data_i = real_dose.data[i_batch,:,:]
    real_structure_masks_data_i = real_structure_masks.data[i_batch,:,:]

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(real_ct_data_i, cmap='gray', origin='lower')
    plt.title("CT")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(real_possible_dose_mask_data_i, cmap='gray', origin='lower')
    plt.title("Possible dose mask")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(real_structure_masks_data_i,
              cmap='gray', origin='lower')
    plt.title("Structure masks")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(real_dose_data_i, cmap='gray', origin='lower')
    plt.title("Dose")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(gen_img[i_batch,0,:,:], cmap='gray', origin='lower')
    plt.title("Predicted dose")
    plt.axis('off')

    plt.savefig(f'{path_save_fig}_{n_epoch}.png')
    plt.show()

def sample_images(epoch, dataloader, generator, tensor_type):
    imgs = next(iter(dataloader))
    fake_dose = generator(imgs["img"].type(tensor_type))
    save_image(imgs, fake_dose.data.cpu(), epoch, f"./images/generator/epoch")