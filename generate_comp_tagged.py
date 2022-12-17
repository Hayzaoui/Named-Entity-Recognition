from torch import save, load
from torch.utils.data import TensorDataset, DataLoader
from gensim import downloader
import numpy as np
from torch import nn, Tensor
from main import features
from sklearn import metrics
from FNN import test
from FNN import predict
import pickle

GLOVE_PATH = 'glove-twitter-200'
glove = downloader.load(GLOVE_PATH)
#print(type(glove))

def writes_tagged_test(pred, test_path, predictions_path, list_of_words):
    data_path = os.path.join('data', test_path)
    output_file = open(predictions_path, "w+", encoding='utf-8')
    assert len(list_of_words) == len(pred)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == "\t\n" or line == "\n":
                output_file.write(line)
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            word = line
            # if word != list_of_words[index]:
            #     print(f"{word} != {list_of_words[index]}")
            output_file.write(f"{word}\t{pred[index]}\n")
            index += 1
    output_file.close()
def main():
    batch_size = 300



    model2 = load("model2.plk")



    vecs_test = []
    test_path = 'data/test.untagged'

    with open(test_path, 'r', encoding='utf-8-sig') as f:
        for lines in f.readlines():

            if lines != ['\n']:
                line = lines
                real_word = line[0]
                word = line[0].lower()

                if word not in glove.key_to_index:
                    vecs_test.append(features(real_word, np.zeros(200)))
                else:
                    vecs_test.append(features(real_word, glove[word]))


    tensor_x_test = Tensor(np.array(vecs_test))  # transform to torch tensor

    test_dataset = TensorDataset(tensor_x_test)  # create your datset
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)  # create your dataloader
    predictions = predict(model2, test_loader)

    output_file = open('prediction.tagged', "w+", encoding='utf-8')
    index = 0
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == "\t\n" or line == "\n":
                output_file.write(line)
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            word = line
            # if word != list_of_words[index]:
            #     print(f"{word} != {list_of_words[index]}")
            output_file.write(f"{word}\t{predictions[index]}\n")
            index += 1
    output_file.close()



if __name__ == '__main__':
    main()
