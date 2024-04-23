import torch.nn as nn
import torch
from tqdm import tqdm


def get_device():
    # Local apple silicon mps
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def count_parameters(model):
    model_params_str, model_params = "", 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params_str = (
                model_params_str + f"{name}/{param.size()}/{param.numel()}|"
            )
            model_params += param.numel()
    return model_params_str, model_params


def custom_collate_fn(batch, device):
    """
    Dimension key:
     - B: Batch size
     - L: Length of sequence
     - C: Number of classes
    """
    # Prepare the datapoints
    x, y = zip(*batch)

    x_tensor = [torch.LongTensor(xi) for xi in x]
    x_tensor_BL = torch.stack(x_tensor)

    y_tensor = [torch.FloatTensor(yi) for yi in y]
    y_tensor_BC = torch.stack(y_tensor)

    lengths = [len(label) for label in y]
    lengths_tensor_B = torch.LongTensor(lengths)

    return x_tensor_BL.to(device), y_tensor_BC.to(device), lengths_tensor_B.to(device)


class ImageMulticlassClassifierBaseline(nn.Module):

    def __init__(self, hyperparameters, num_classes=10):
        super(ImageMulticlassClassifierBaseline, self).__init__()

        self.input_dim = hyperparameters["input_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]

        self.image_embedding = nn.Flatten()
        if self.freeze_embeddings:
            self.embedding.weight.requires_grad_ = False
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()
        )
        self.output_layer = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, inputs_BL):
        """
        Dimension key:
         - B: Batch size
         - E: Size of embedding
         - W: Size of word window embedding
         - H: Size of hidden layer embedding
         - L: Length of sequence
         - C: Number of classes
        """
        print(f"inputs_BL.size(): {inputs_BL.size()}")
        B, L = inputs_BL.size()

        embedded_images_BLE = self.image_embedding(inputs_BL)
        # For debugging
        # print(f"embedded_images_BLE.size(): {embedded_images_BLE.size()}")
        hidden1_BH = self.hidden_layer1(embedded_images_BLE)
        hidden2_BH = self.hidden_layer2(hidden1_BH)
        output_BC = self.output_layer(hidden2_BH)
        softmax_BC = nn.functional.softmax(output_BC, dim=1)
        return softmax_BC


def ce_loss_function(batch_outputs, batch_labels):
    # :== Do not remove. Used for debugging
    # print(batch_outputs)
    # print(batch_outputs.shape)
    # print(batch_labels)
    # print(batch_labels.shape)
    # :== Do not remove. Used for debugging

    # Calculate the loss for the whole batch
    celoss = nn.CrossEntropyLoss()
    loss = celoss(batch_outputs, batch_labels)

    return loss


def train_epoch(loss_function, optimizer, model, loader, device):
    """
    Dimension key:
     - B: Batch size
     - L: Length of sequence
     - C: Number of classes
    """

    # Keep track of the total loss for the batch
    total_loss = 0
    for batch_inputs_BL, batch_labels_BC in loader:
        batch_inputs_BL = batch_inputs_BL.to(device)
        batch_labels_BC = batch_labels_BC.to(device)
        images = batch_inputs_BL.view(batch_inputs_BL.shape[0], -1)
        # print(batch_inputs) # for debugging
        # Clear the gradients
        optimizer.zero_grad()
        # Run a forward pass
        # print("batch_inputs_BL.size()")
        # print(batch_inputs_BL.size())
        outputs_BC = model.forward(images)
        # print("outputs_BC.size()")
        # print(outputs_BC.size())
        # Compute the batch loss
        # print("batch_labels_BC.size()")
        # print(batch_labels_BC.size())
        loss = loss_function(outputs_BC, batch_labels_BC)
        # Calculate the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()
        # print("batch_lengths_B.size()")
        # print(batch_lengths_B.size())
        total_loss += loss.item()

    return total_loss


def train(loss_function, optimizer, model, loader, num_epochs=10000):
    epoch_and_loss_list = [["epoch", "loss"]]
    for epoch in range(num_epochs):
        print(f"epoch={epoch}")
        epoch_loss = train_epoch(loss_function, optimizer, model, loader)
        if epoch % 10 == 0:
            print(f"epoch={epoch}, epoch_loss={epoch_loss}")
            epoch_and_loss_list.append([epoch, float(epoch_loss.numpy())])
    return epoch_and_loss_list
