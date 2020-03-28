import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data.dataset import Dataset
from seq_data import getTrainData


class DatasetFromCSV(Dataset):
    def __init__(self,mode):
        '''
        mode: train or test
        '''
        X, Y , x_test, y_test= getTrainData(128)
        self.input=X if mode=='train' else x_test
        self.output=Y if mode=='train' else y_test
 
    def __getitem__(self, index):
        inputs=torch.LongTensor(self.input[index])
        outputs=torch.LongTensor(self.output[index])
        return inputs,outputs
 
    def __len__(self):
        return len(self.input)

class SimpleEncoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleEncoder, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=configure["INPUT_SIZE"], 
                                            embedding_dim=configure["EMBEDDING_DIM"])
 
        self.gru = torch.nn.LSTM(input_size=configure["EMBEDDING_DIM"],
                                hidden_size=configure['HIDDEN_SIZE'],
                                num_layers=configure["NUM_LAYERS"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure['HIDDEN_SIZE'], configure["INPUT_SIZE"])



    def forward(self, input):
        
        # Embedding
        embedding = self.embedding(input)
        print(embedding)
        # Call the GRU
        out, hidden = self.gru(embedding)

        return out, hidden

class Attention(nn.Module):
    """ Simple Attention

    This Attention is learned from weight
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

        # Declare the Attention Weight
        self.W = nn.Linear(dim, 1)

        # Declare the coverage feature
        self.coverage_feature = nn.Linear(1,dim)

    def forward(self, output, context, coverage):

        # declare the size
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # Expand the output to the num of timestep
        output_expand = output.expand(batch_size, input_size, hidden_size)

        # reshape to 2-dim
        output_expand = output_expand.reshape([-1, hidden_size])
        context = context.reshape([-1, hidden_size])

        # transfer the coverage to features
        coverage_feature = self.coverage_feature(coverage.reshape(-1,1))

        # Learning the attention
        attn = self.W(output_expand + context + coverage_feature)
        attn = attn.reshape(-1, input_size)
        attn = F.softmax(attn, dim=1)
        
        # update the coverage
        coverage = coverage + attn

        context = context.reshape(batch_size, input_size, hidden_size)
        attn = attn.reshape(batch_size, -1, input_size)

        # get the value of a
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn, coverage

class SimpleDecoder(torch.nn.Module):

    def __init__(self, configure):
        super(SimpleDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["INPUT_SIZE"], 
                                            embedding_dim=configure["EMBEDDING_DIM"])

        self.gru = torch.nn.GRU(input_size=configure["EMBEDDING_DIM"],
                                hidden_size=configure['HIDDEN_SIZE'],
                                num_layers=configure["NUM_LAYERS"],
                                bidirectional=False,
                                batch_first=True)

        self.fc = torch.nn.Linear(configure['HIDDEN_SIZE'], configure["INPUT_SIZE"])



    def forward(self, input, hidden):

        # Embedding
        embedding = self.embedding(input)

        # Call the GRU
        out, hidden = self.gru(embedding, hidden)

        out = self.fc(out.view(out.size(0),-1))

        return out, hidden

class AttentionDecoder(torch.nn.Module):
    
    def __init__(self, configure, device):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.configure = configure
        self.device = device
        self.configure = configure

        # Embedding
        self.embedding = torch.nn.Embedding(num_embeddings=configure["INPUT_SIZE"], 
                                            embedding_dim=configure["EMBEDDING_DIM"])

        self.gru = torch.nn.LSTM(input_size=configure["EMBEDDING_DIM"]+configure['HIDDEN_SIZE'],
                                hidden_size=configure['HIDDEN_SIZE'],
                                num_layers=configure["NUM_LAYERS"],
                                bidirectional=False,
                                batch_first=True)

        self.att = Attention(configure['HIDDEN_SIZE'])

        self.fc = torch.nn.Linear(configure['HIDDEN_SIZE'], configure["INPUT_SIZE"])

        self.p = torch.nn.Linear(configure["MAX_OUTPUT"]+configure["EMBEDDING_DIM"], 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input, hidden, encoder_output, z, content, coverage):

        # Embedding
        embedding = self.embedding(input)
        # print(embedding.squeeze().size())

        combine = torch.cat([embedding,z],2)
        # print(combine.squeeze().size())
        # Call the GRU
        out, hidden = self.gru(combine, hidden)

        # call the attention
        output, attn, coverage = self.att(output=out, context=encoder_output, coverage=coverage)
        

        index = content
        attn = attn.view(attn.size(0),-1)
        attn_value = torch.zeros([attn.size(0), self.configure["num_words"]]).to(self.device)
        attn_value = attn_value.scatter_(1, index, attn)

        out = self.fc(output.view(output.size(0),-1))
        # print(torch.cat([embedding.squeeze(), combine.squeeze()], 1).size(), )
        p = self.sigmoid(self.p(torch.cat([embedding.squeeze(), combine.squeeze()], 1)))
        # print(p)
        out = (1-p)*out + p*attn_value
        # print(attn_value.size(), output.size())

        return out, hidden, output, attn, coverage

if __name__ == "__main__":
    # Declare the hyperparameter 定义超参数
    CONFIGURE={
        "EPOCHS":100,
        "BATCH_SIZE":128,
        "INPUT_SIZE":5,
        "HIDDEN_SIZE":128,
        "EMBEDDING_DIM":5,
        "LEARNING_RATE":0.01,
        "MAX_CONTENT":500,
        "MAX_OUTPUT":10,
        "NUM_LAYERS":1
    }
    DEVICE = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
    # Declare the encoder model
    model_encoder = SimpleEncoder(CONFIGURE).to(DEVICE)
    model_decoder = AttentionDecoder(CONFIGURE, DEVICE).to(DEVICE)


    # Define the optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    # encoder optimizer
    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=CONFIGURE["LEARNING_RATE"])
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=CONFIGURE["LEARNING_RATE"])

    train_data=DatasetFromCSV('train')
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=CONFIGURE['BATCH_SIZE'])
    test_data=DatasetFromCSV('test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=CONFIGURE['BATCH_SIZE'])
    print(train_loader)
    # Training
    for epoch in range(CONFIGURE["EPOCHS"]):
        for idx, item in enumerate(train_loader):
            print(idx,item)
            # transfer to long tensor
            inputs, target = [i.type(torch.LongTensor).to(DEVICE) for i in item]
            print(inputs)
            if inputs.size(0) != CONFIGURE["BATCH_SIZE"]: continue
            # Encoder   
            encoder_out, encoder_hidden = model_encoder(inputs)
            # Decoder 
            # declare the first input of decoder
            decoder_input = torch.tensor([0]*CONFIGURE['BATCH_SIZE'], 
                                         dtype=torch.long, device=DEVICE).view(CONFIGURE['BATCH_SIZE'], -1)
            decoder_hidden = encoder_hidden
            z = torch.ones([CONFIGURE['BATCH_SIZE'],1,CONFIGURE['HIDDEN_SIZE']]).to(DEVICE)
            coverage = torch.zeros([CONFIGURE['BATCH_SIZE'],CONFIGURE["MAX_CONTENT"]]).to(DEVICE)
            seq_loss = 0
            for i in range(configure["MAX_OUTPUT"]):

                decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, inputs, coverage)

                coverage = coverage

                if random.randint(1, 10) > 5:
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(CONFIGURE['BATCH_SIZE'], -1)
                else:
                    decoder_input = target[:,i].view(CONFIGURE['BATCH_SIZE'], -1)

                decoder_hidden = decoder_hidden

                step_coverage_loss = torch.sum(torch.min(attn.reshape(-1,1), coverage.reshape(-1,1)), 1) 
                step_coverage_loss = torch.sum(step_coverage_loss)
                # print(coverage)
                # print("---")
                # decoder_output = decoder_output.reshape(configure["batch_size"], -1, 1)
                # print(step_coverage_loss)
                # print((criterion(decoder_output, target[:,i].reshape(configure["batch_size"],-1))))
                # print(-torch.log(decoder_output+target[:,i]))
                seq_loss += (criterion(decoder_output, target[:,i]))

                # print(seq_loss)
        
                seq_loss += step_coverage_loss
      
                # print(decoder_input)
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            seq_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            if (idx) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Coverage Loss: {:4f}' 
                    .format(epoch+1, CONFIGURE["EPOCHS"], idx, len(train_loader), seq_loss.item(),step_coverage_loss.item()))


        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for idx, item in enumerate(test_loader):
                
                # transfer to long tensor
                inputs, target = [i.type(torch.LongTensor).to(device) for i in item]
                
                if inputs.size(0) != CONFIGURE['BATCH_SIZE']: continue
                # Encoder   
                encoder_out, encoder_hidden = model_encoder(inputs)
                
                # Decoder 
                # declare the first input of decoder
                decoder_input = torch.tensor([0]*CONFIGURE['BATCH_SIZE'], 
                                            dtype=torch.long, device=DEVICE).view(CONFIGURE['BATCH_SIZE'], -1)
                decoder_hidden = encoder_hidden
                seq_loss = 0
                result = []
                z = torch.ones([CONFIGURE['BATCH_SIZE'],1,CONFIGURE['HIDDEN_SIZE']]).to(DEVICE)
                coverage = torch.zeros([CONFIGURE['BATCH_SIZE'],CONFIGURE["MAX_CONTENT"]]).to(DEVICE)
                for i in range(CONFIGURE["MAX_OUTPUT"]):
                    decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, inputs, coverage)

    
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(CONFIGURE["BATCH_SIZE"], -1)


                    decoder_hidden = decoder_hidden

                    total += CONFIGURE["BATCH_SIZE"]
                    correct += (torch.max(decoder_output, 1)[1] == target[:,i]).sum().item()
                    # print(torch.max(decoder_output, 1)[1],target[:,i])
                    result.append(index_word[torch.max(decoder_output, 1)[1][1].item()])
                
            with open("test.txt", "a+", encoding="utf-8") as a: a.write("".join(result)+"\n")

            print('Test Accuracy of the model on the test: {} %'.format(100 * correct / total)) 
