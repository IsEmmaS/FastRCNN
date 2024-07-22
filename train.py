import torch

from dataloader import device

print(f'Training on {device}')


def train_batch(inputs, model, optimizer):
    """
    Train a model in one Batch
    """
    model.train()
    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

    optimizer.zero_grad()
    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()

    return loss, losses


@torch.no_grad()
def validate_batch(inputs, model, optimizer):
    model.train()

    input, targets = inputs
    input = list(image.to(device) for image in input)
    targets = [{key: value.to(device) for key, value in t.items()} for t in targets]

    optimizer.zero_grad()

    losses = model(input, targets)
    loss = sum(loss for loss in losses.values())
    return loss, losses


if __name__ == '__main__':
    from torch_snippets import Report
    from model import get_model, device
    from dataloader import train_dataloader, test_dataloader

    model = get_model()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=.005,
                                momentum=.9,
                                weight_decay=.0005)

    num_epochs = 20
    log = Report(n_epochs=num_epochs)

    for epoch in range(num_epochs):
        _n = len(train_dataloader)
        for index, inputs in enumerate(train_dataloader):
            loss, losses = train_batch(inputs, model, optimizer)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                [losses[key] for key in {'loss_classifier',
                                         'loss_box_reg',
                                         'loss_objectness',
                                         'loss_rpn_box_reg'}]
            pos = (epoch + (index + 1) / _n)

            log.record(pos,
                       trn_loss=loss.item(),
                       trn_loc_loss=loc_loss.item(),
                       trn_regr_loss=regr_loss.item(),
                       trn_objectness_loss=loss_objectness.item(),
                       trn_rpn_box_reg_loss=loss_rpn_box_reg.item()
                       )
        _n = len(test_dataloader)
        for index, inputs in enumerate(test_dataloader):
            loss, losses = validate_batch(inputs, model, optimizer)
            loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                [losses[key] for key in {'loss_classifier',
                                         'loss_box_reg',
                                         'loss_objectness',
                                         'loss_rpn_box_reg'}]
            pos = (epoch + (index + 1) / _n)

            log.record(pos,
                       val_loss =loss.item(),
                       val_loc_loss=loc_loss.item(),
                       val_regr_loss=regr_loss.item(),
                       val_objectness_loss=loss_objectness.item(),
                       val_rpn_box_reg_loss=loss_rpn_box_reg.item()
                       )
        log.report_avgs(epoch + 1)

    log.plot_epochs(['trn_loss', 'val_loss'])
    torch.save(model.state_dict(), f'weights/model.pth')
