import dataset_interface
import pdb
import torch


from dataset_interface import MyDataset, to_depth, to_disparity



def main():
    dataset = MyDataset("train")

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    i = 0
    for tup in loader:
        i+=1
        print(f"I have completed {i} iterations out of {len(loader)}")
    


if __name__ == "__main__":
    main()