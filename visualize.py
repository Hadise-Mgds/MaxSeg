import matplotlib.pyplot as plt

def display_volume_slice(image_volume, segmentation_mask, slice_index=150):
    """
    Displays a 2D slice from the 3D image volume and its corresponding segmentation mask.

    Args:
        image_volume (numpy.ndarray): A 5D tensor with shape [batch, channel, depth, height, width].
        segmentation_mask (numpy.ndarray): Corresponding ground truth mask with the same shape.
        slice_index (int): The index along the depth axis to visualize.
    """
    
    # Plot the image slice
    plt.subplot(1, 2, 1)
    plt.imshow(image_volume[1, 0, :, :, slice_index], cmap='gray')
    plt.title("Input Image Slice")
    plt.axis("off")

    # Plot the mask slice
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask[1, 0, :, :, slice_index], cmap='gray')
    plt.title("Segmentation Mask Slice")
    plt.axis("off")

    # Display both plots side by side
    plt.tight_layout()
    plt.show()
