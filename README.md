# Project Overview

## Project Title: Fine-Grained Classification Using Vision Transformers

### Objective:
The objective of this project is to evaluate the performance of various pre-trained Vision Transformer (ViT) models for fine-grained image classification tasks. By using transfer learning on these models on a specific dataset, we aim to identify the most effective vision transformer model for this type of classification.

### Background and Importance:

In recent years, the field of computer vision has seen significant advancements due to the development of deep learning models. Convolutional Neural Networks (CNNs) have been the backbone of many successful computer vision applications. However, a new paradigm called Vision Transformers (ViTs) has emerged, demonstrating superior performance on a variety of image classification tasks.

Fine-grained image classification is a challenging problem that involves distinguishing between very similar categories within a broader class. Examples include differentiating between species of birds, types of flowers, or models of cars. These tasks are crucial in various domains such as biodiversity monitoring, agriculture, healthcare, and manufacturing.

## Project Plan

### Phase 1: Setup and Data Preparation

1. **Setup Environment**
   - Install necessary libraries: PyTorch, `timm`, torchvision, and other dependencies.
   - Set up a version control system (e.g., Git) for project tracking.

2. **Data Collection and Preprocessing**
   - Collect and clean the fine-grained classification dataset.
   - Split the dataset into training, validation, and test sets.
   - Apply necessary transformations (e.g., resizing, normalization).

### Phase 2: Model Selection and Fine-Tuning

3. **Model Selection**
   - Select pre-trained models from the `timm` library:
     - DeiT: `deit_tiny_patch16_224`, `deit_small_patch16_224`, `deit_base_patch16_224`
     - Swin Transformer: `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224`, `swin_base_patch4_window7_224`
     - Vanilla ViT:  `vit_base_patch16_224`

4. **Transfer Learning Process**
   - Load each pre-trained model.
   - Replace the final classification layer to match the number of classes in the fine-grained dataset.
   - Define loss function, optimizer, and learning rate scheduler.
   - Train each model on the training set while validating on the validation set.
   - Save the final model weights.

### Phase 3: Evaluation and Inference

5. **Model Evaluation**
   - Evaluate each fine-tuned model on the test set.
   - Calculate performance metrics: accuracy, precision, recall, F1-score.
   - Compare the performance of all models.

### Checkpoints
Below are the checkpoints of the models after training has been completed.

| Checkpoints              | 
| :---------------- | 
| [ViT Base](https://drive.google.com/file/d/1DiD1tZ9u3aYrlOPlX_C2UicPVnBPZG4E/view?usp=share_link)        | 
| [DeiT Base](https://drive.google.com/file/d/1dqMvI_su8JLLjfzXZyySkawejeHuLoQ2/view?usp=share_link)          |  
| [DeiT Small](https://drive.google.com/file/d/1VmVbl2y1iMCsAGp9ckGc4aNOFbD9BinR/view?usp=share_link)    | 
| [DeiT Tiny](https://drive.google.com/file/d/14eH_SbVb5mPuNneznzNw06s4CSTyvKAD/view?usp=share_link) |  
| [Swin Base](https://drive.google.com/file/d/150YtaDCujbiJGhDIrWGJ2Z58GDfQsQDh/view?usp=share_link) |  
| [Swin Small](https://drive.google.com/file/d/1XDTTaa2mjAedSzzrhXd9_QJso3C5QBMr/view?usp=share_link) |  
| [Swin Tiny](https://drive.google.com/file/d/1lQc3InmEeHpdj3rIhi6XkL9NuU53-vQA/view?usp=share_link) |  

## Conclusion

This project aims to leverage the power of Vision Transformers for fine-grained classification by systematically fine-tuning and evaluating multiple pre-trained models. The results will provide insights into the most effective ViT models for such tasks, potentially informing future work and applications in this domain.
