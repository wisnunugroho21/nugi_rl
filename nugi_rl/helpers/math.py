import torch
from torch import Tensor


def normalize(
    data: Tensor,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    clip: Tensor | None = None,
) -> Tensor:
    if mean is not None and std is not None:
        data_normalized = (data - mean) / (std + 1e-8)
    else:
        data_normalized = (data - data.mean()) / (data.std() + 1e-8)

    if clip:
        data_normalized = data_normalized.clamp(-1 * clip, clip)

    return data_normalized


def prepro_half(image: Tensor) -> Tensor:
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image


def prepro_crop(image: Tensor) -> Tensor:
    image = image[35:195]
    return image


def prepo_full(image: Tensor) -> Tensor:
    image = image[35:195]  # crop
    image = image[:, :, 0]
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image


def prepo_full_one_dim(image: Tensor) -> Tensor:
    image = prepo_full(image)
    image = image.float().flatten()
    return image


def prepro_half_one_dim(image: Tensor) -> Tensor:
    image = prepro_half(image)
    image = image.float().flatten()
    return image


def prepo_crop(image: Tensor) -> Tensor:
    image = image[35:195]  # crop
    return image


def new_std_from_rewards(rewards: Tensor, reward_target: Tensor) -> Tensor:
    mean_reward = (reward_target - rewards).mean()
    new_std = mean_reward / reward_target

    new_std = torch.where(new_std < 0.25, 0.25, new_std)
    new_std = torch.where(new_std > 1.0, 1.0, new_std)

    return new_std


def count_new_mean(prevMean: Tensor, prevLen: Tensor, newData: Tensor) -> Tensor:
    return ((prevMean * prevLen) + newData.sum(0)) / (prevLen + newData.shape[0])


def count_new_std(prevStd: Tensor, prevLen: Tensor, newData: Tensor) -> Tensor:
    return (
        ((prevStd.pow(2) * prevLen) + (newData.var(0) * newData.shape[0]))
        / (prevLen + newData.shape[0])
    ).sqrt()
