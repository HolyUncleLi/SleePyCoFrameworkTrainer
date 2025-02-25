import sklearn_crfsuite

EPOCH = 100

BATCH_SIZE = 12
seq_len = 20


def TrainCRF(dataloader, model):

    # CRF模型
    crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.25, c2=0.018, max_iterations=100,
                                     all_possible_transitions=True, verbose=True)

    features = []
    labels = []
    for index, (x, y) in enumerate(dataloader):
        label, out = model(x.view(BATCH_SIZE * seq_len, 1, 3000).transpose(1, 2).cuda())
        # print(index, x.shape, y.shape, label.shape, out.shape)

        for i in range(BATCH_SIZE):
            sequence = []
            label = []
            for j in range(seq_len):
                hidden = {f'feature_{k}': out[i, j, k].cpu().detach().numpy() for k in range(5)}
                sequence.append(hidden)
                label.append(f'{y[i,j].cpu().detach().numpy()}')
            features.append(sequence)
            labels.append(label)
    print('__..__')
    print(len(features), len(labels))
    print(len(features[0]), len(labels[0]))
    print(features[0][0], labels[0][0])

    crf_model.fit(features, labels)

    # 打印转移矩阵
    transition_features = crf_model.transition_features_
    for (label_from, label_to), weight in transition_features.items():
        print(f"Transition from {label_from} to {label_to}: {weight}")
