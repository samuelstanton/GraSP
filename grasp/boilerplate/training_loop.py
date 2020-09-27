from torch import nn, optim
from grasp.boilerplate import train_epoch, test_epoch
from grasp.utils.common_utils import PresetLRScheduler


def training_loop(net, trainloader, testloader, learning_rate, weight_decay, num_epochs, s3_logger):
    s3_logger.add_table('train_metrics')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_schedule = {0: learning_rate,
                   int(num_epochs * 0.5): learning_rate * 0.1,
                   int(num_epochs * 0.75): learning_rate * 0.01}
    lr_scheduler = PresetLRScheduler(lr_schedule)
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, trainloader, optimizer, criterion, lr_scheduler, epoch)
        test_loss, test_acc = test_epoch(net, testloader, criterion)
        s3_logger.log(dict(
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            lr=lr_scheduler.get_lr(optimizer)
        ), step=epoch + 1, table_name='train_metrics')
        s3_logger.write_csv()
