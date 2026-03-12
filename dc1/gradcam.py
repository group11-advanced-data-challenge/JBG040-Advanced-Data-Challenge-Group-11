import torch
import torch.nn.functional as F


class GradCAM:
    """
    Standard Grad-CAM utility for CNN classifiers.

    Features:
    - stores activations from a target convolutional layer using a forward hook
    - stores gradients from the same layer using a backward hook
    - supports explaining:
        1) the predicted class
        2) any manually specified class
        3) both predicted class and true class
    - dynamically upsamples the CAM to match the input image size
    - supports hook cleanup via remove_hooks()
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input_tensor, output_tensor):
        """
        Save forward activations from the target layer.
        Expected shape: [B, C, H, W]
        """
        self.activations = output_tensor

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Save backward gradients from the target layer.
        grad_output[0] usually has shape: [B, C, H, W]
        """
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """
        Remove hooks to avoid accumulation when creating GradCAM multiple times.
        """
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def _compute_cam_from_current_state(self, input_image):
        """
        Compute Grad-CAM from the currently stored activations and gradients.

        Args:
            input_image: torch.Tensor of shape [1, C, H, W]

        Returns:
            cam: numpy array of shape [H, W]
        """
        if self.activations is None:
            raise RuntimeError("Activations are None. The forward hook may not have fired.")
        if self.gradients is None:
            raise RuntimeError("Gradients are None. The backward hook may not have fired.")

        # Single-image batch
        activations = self.activations[0]   # [C, h, w]
        gradients = self.gradients[0]       # [C, h, w]

        # Channel-wise global average pooling -> weights
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted sum of activations
        cam = torch.zeros_like(activations[0])  # [h, w]
        for channel_index, weight in enumerate(weights):
            cam += weight * activations[channel_index]

        # Keep only positive contributions
        cam = F.relu(cam)

        # Safe normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample dynamically to input size
        _, _, input_height, input_width = input_image.shape
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # [1, 1, h, w]
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()
        return cam

    def generate(self, input_image, class_idx=None):
        """
        Generate Grad-CAM for one class.

        Args:
            input_image: torch.Tensor of shape [1, C, H, W]
            class_idx: int or None
                - None: explain the predicted class
                - int: explain the specified class

        Returns:
            cam: numpy array [H, W]
            class_idx: explained class index
            logits: torch.Tensor [1, num_classes]
            probs: torch.Tensor [1, num_classes]
        """
        self.model.eval()

        if input_image.dim() != 4 or input_image.size(0) != 1:
            raise ValueError("input_image must have shape [1, C, H, W].")

        self.model.zero_grad()

        logits = self.model(input_image)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        cam = self._compute_cam_from_current_state(input_image)
        return cam, class_idx, logits.detach(), probs.detach()

    def generate_pred_true(self, input_image, true_class_idx):
        """
        Generate both predicted-class CAM and true-class CAM.

        Args:
            input_image: torch.Tensor of shape [1, C, H, W]
            true_class_idx: int

        Returns:
            dict with:
                - pred_cam
                - true_cam
                - pred_class
                - true_class
                - logits
                - probs
        """
        self.model.eval()

        if input_image.dim() != 4 or input_image.size(0) != 1:
            raise ValueError("input_image must have shape [1, C, H, W].")

        self.model.zero_grad()
        logits = self.model(input_image)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()

        # Predicted-class CAM
        self.model.zero_grad()
        pred_score = logits[:, pred_class]
        pred_score.backward(retain_graph=True)
        pred_cam = self._compute_cam_from_current_state(input_image)

        # True-class CAM
        self.model.zero_grad()
        true_score = logits[:, true_class_idx]
        true_score.backward(retain_graph=True)
        true_cam = self._compute_cam_from_current_state(input_image)

        return {
            "pred_cam": pred_cam,
            "true_cam": true_cam,
            "pred_class": pred_class,
            "true_class": true_class_idx,
            "logits": logits.detach(),
            "probs": probs.detach(),
        }