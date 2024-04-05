from torch_lr_finder import LRFinder
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="mps")
lr_finder.range_test(train_loader,end_lr=1, num_iter=100, step_mode="exp")
lr_finder.plot(log_lr=False)
lr_finder.reset()