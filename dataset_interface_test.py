import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset



def main():
    dataset = MyDataset("test")

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for tup in loader:
        pdb.set_trace()


if __name__ == "__main__":
    main()