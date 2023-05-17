# Hyperparams
num_classes = 10
epochs = 10
batch_size = 7
learning_rate = 1e-5
log_interval = 20
sample_size = 128
sample_duration = 16


# Train
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    
    train_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=True, transform=transform)
    val_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # Create model
    model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)

    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    for epoch in range(epochs):
        # Train the model
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_epoch(model, criterion, val_loader, device, epoch, logger, writer)

        # Save model
        # torch.save(model.state_dict(), os.path.join(model_path, "slr_cnn3d_epoch{:03d}.pth".format(epoch+1)))
        torch.save(model, os.path.join(model_path, "slr_cnn3d_epoch{:03d}.pth".format(epoch+1)))