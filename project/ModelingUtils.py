import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt

from unsupervised.MonodepthUtils import reconstruct_input_from_disp_maps, unsupervised_monodepth_loss

TRAIN_REPORT_INTERVAL = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def unsupervised_loss(tup, model: nn.Module):
    """

    :param tup: A tuple from the dataloader
    :return: loss
    """

    left_img, right_img, _, _ = tup

    left_img = left_img.to(DEVICE)
    right_img = right_img.to(DEVICE)

    stereo_pair = (left_img, right_img)
    disp_maps = model.forward(left_img)

    reconstructions = reconstruct_input_from_disp_maps(stereo_pair, disp_maps)

    loss = unsupervised_monodepth_loss(stereo_pair, disp_maps, reconstructions)

    return loss


def train(train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            model: nn.Module,
            model_savedir: str,
            tbx_log_dir: str,
            initial_lr: float = 1e-4,
            num_epochs: int = 2,
            supervised: bool = False):

    if supervised:
        raise NotImplementedError("Implement this later!")

    model = model.to(DEVICE)
    model.train()
    best_val_loss = float("inf")

    tbx_writer = SummaryWriter(log_dir=tbx_log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

    train_tbx_idx = 0
    for epoch in tqdm(range(num_epochs)):
        tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], train_tbx_idx)

        num_train_examples = 0
        running_loss = 0
        for tup in tqdm(train_loader, desc=f"Training - Epoch {epoch}", leave=False):
            examples_in_batch = tup[0].shape[0]

            optimizer.zero_grad()
            loss = unsupervised_loss(tup, model)
            running_loss += examples_in_batch * loss.item()
            loss.backward()
            optimizer.step()

            num_train_examples += examples_in_batch
            train_tbx_idx += examples_in_batch
            if num_train_examples > TRAIN_REPORT_INTERVAL:
                running_loss /= num_train_examples
                tbx_writer.add_scalar("train/loss", running_loss, train_tbx_idx)

                running_loss = 0
                num_train_examples = 0

        val_loss = 0
        total_val_examples = 0
        for tup in tqdm(val_loader, desc=f"Validation - Epoch {epoch}", leave=False):
            with torch.no_grad():
                examples_in_batch = tup[0].shape[0]
                val_loss += examples_in_batch * unsupervised_loss(tup, model).item()
                total_val_examples += examples_in_batch
        val_loss /= total_val_examples

        tbx_writer.add_scalar("val/loss", val_loss, epoch)

        torch.save(model, f"{model_savedir}/last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"{model_savedir}/best.pt")

        lr_scheduler.step()

    print(f"Training completed! Best validation loss was {best_val_loss}")


def test(test_loader: torch.utils.data.DataLoader, model: nn.Module):
    model = model.to(DEVICE)
    model.eval()

    test_loss = 0
    total_test_examples = 0
    for tup in tqdm(test_loader):
        with torch.no_grad():
            examples_in_batch = tup[0].shape[0]
            test_loss += examples_in_batch * unsupervised_loss(tup, model).item()
            total_test_examples += examples_in_batch
    print(f"Average test loss was {test_loss/total_test_examples}")
    # TODO: add a function to compare this to ground-truth depth using methods to convert from disparity to depth

# TODO: add a function that can display generated disp maps for images next to GT depth maps and the original images
def visualize_disparity_maps(data_loader: torch.utils.data.DataLoader, model: nn.Module, savepath: str):
    model.to(DEVICE)
    model.eval()

    for tup in data_loader:
        data_tuple_to_plt_image(tup, model)
        plt.savefig(savepath)
        break  # TODO: refactor this to avoid this hacky logic to get access to data


def data_tuple_to_plt_image(tup, model: nn.Module):
    left_image, right_image, left_depth_gt, right_depth_gt = tup
    left_image = left_image.to(DEVICE)
    right_image = right_image.to(DEVICE)

    with torch.no_grad():
        left_to_right_disp, right_to_left_disp = model.forward(left_image)

        recons = reconstruct_input_from_disp_maps((left_image, right_image), (left_to_right_disp, right_to_left_disp))

    left_image_np = left_image.permute((0, 2, 3, 1))[0, :, :, :].cpu().detach().numpy()
    left_disp_np = left_to_right_disp[0, :, :].cpu().detach().numpy()
    left_depth_gt_np = left_depth_gt[0, :, :].cpu().detach().numpy()
    left_recon_np = recons[0].permute((0, 2, 3, 1))[0, :, :, :].cpu().detach().numpy()

    fig = plt.figure(figsize=(21, 7))

    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(left_image_np)
    plt.axis('off')
    plt.title("Left Image")

    fig.add_subplot(rows, cols, 2)
    plt.imshow(left_depth_gt_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")

    fig.add_subplot(rows, cols, 3)
    plt.imshow(left_recon_np)
    plt.axis('off')
    plt.title("Reconstructed Left Image")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(left_disp_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Predicted Disparity Map")

    return
