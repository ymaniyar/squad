# dep_parser.py 

import spacy
import json

def main():
    nlp = spacy.load('en_core_web_sm')
    # doc = nlp(u"Autonomous cars shift insurance liability towards manufacturers")
    # for token in doc: 
    #     print("token: ", token)
    #     print("head: ", token.head.text)
    #     print("dependency: ", token.dep_)
    #     print("children: ", [child for child in token.children])

    # doc = nlp("--OOV-- During the Cold War between the Soviet Union and the United States , huge stockpiles of uranium were amassed and tens of thousands of nuclear weapons were created using enriched uranium and plutonium made from uranium . Since the break - up of the Soviet Union in 1991 , an estimated 600 short tons ( 540 metric tons ) of highly enriched weapons grade uranium ( enough to make 40,000 nuclear warheads ) have been stored in often inadequately guarded facilities in the Russian Federation and several other former Soviet states . Police in Asia , Europe , and South America on at least 16 occasions from 1993 to 2005 have intercepted shipments of smuggled bomb - grade uranium or plutonium , most of which was from ex - Soviet sources . From 1993 to 2005 the Material Protection , Control , and Accounting Program , operated by the federal government of the United States , spent approximately US $ 550 million to help safeguard uranium and plutonium stockpiles in Russia . This money was used for improvements and security enhancements at research and storage facilities . Scientific American reported in February 2006 that in some of the facilities security consisted of chain link fences which were in severe states of disrepair . According to an interview from the article , one facility had been storing samples of enriched ( weapons grade ) uranium in a broom closet before the improvement project ; another had been keeping track of its stock of nuclear warheads using index cards kept in a shoe box . --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL--")
    # for token in doc:
    #     print("token: ", token)
    #     print("head: ", token.head.text)
    #     print("dependency: ", token.dep_)
    #     print("children: ", [child for child in token.children])

    x = "--OOV-- The single \" Killer Queen \" from Sheer Heart Attack reached number two on the British charts , and became their first US hit , reaching number 12 on the Billboard Hot 100 . It combines camp , vaudeville , and British music hall with May 's guitar virtuosity . The album 's second single , \" Now I 'm Here \" , a more traditional hard rock composition , was a number eleven hit in Britain , while the high speed rocker \" Stone Cold Crazy \" featuring May 's uptempo riffs is a precursor to speed metal . In recent years , the album has received acclaim from music publications : In 2006 , Classic Rock ranked it number 28 in \" The 100 Greatest British Rock Albums Ever \" , and in 2007 , Mojo ranked it --OOV-- in \" The 100 Records That Changed the World \" . It is also the second of three Queen albums to feature in the book 1001 Albums You Must Hear Before You Die . --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL--"
    y = "--OOV-- Also emerging during this period was New York 's no wave movement , a short - lived art and music scene that began in part as a reaction against punk 's recycling of traditionalist rock tropes and often reflected an abrasive , confrontational and nihilistic worldview . No wave musicians such as the Contortions , Teenage Jesus and the Jerks , Mars , DNA , Theoretical Girls and Rhys Chatham instead experimented with noise , dissonance and atonality in addition to non - rock styles . The former four groups were included on the Eno - produced No New York compilation , often considered the quintessential testament to the scene . The no wave - affiliated label ZE Records was founded in 1978 , and would also produce acclaimed and influential compilations in subsequent years . --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL-- --NULL--"
    # print(len(x))
    # print(len(nlp(x)))
    # # print(len(y))
    # print(len(nlp(y)))
    x_nlp = nlp(x)
    y_nlp = nlp(y)
    x_split = x.split(" ")
    y_split = y.split(" ")
    print("nlp x: ", len(x_nlp))
    print("nlp y: ", len(y_nlp))
    print("x split: ", len(x_split))
    print("y split: ", len(y_split))

    for i in range(0, 358):
        print(i, ": ", x_split[i], ", ", x_nlp[i])

    # print(len(y.split(" ")))
    # print(len(x.split(" ")))
    # for i in range(0, len(nlp(x))):
    #     print(nlp(x)[i], nlp(y)[i])

def createJsonfile():
    with open('./data/word2idx.json') as json_file:
        file = json.load(json_file)
        print(type(file))
        idx2word = {}
        # for line in file:
        #     print(type(line))
        for key in file.keys():
            # print(key, file[key])
            # print(type(file[key]))
            idx2word[file[key]] = key

        # print(len(file), len(idx2word))

        # print(idx2word[2])

    # with open('idx2word.json', 'w') as outfile:
    #     json.dump(idx2word, outfile)




    # dependencies = {}
    # with open('./data/train-v2.0.json') as json_file:
    #     file = json.load(json_file)
    #     for line in file['data']:
    #         # print(line)
    #         paras = line['paragraphs']
    #         for para in paras:
    #             qas = para['qas']
    #             questions = []
    #             for qa in qas: 
    #                 question = qa['question']
    #                 questions.append(question)
    #                 print("question: ", question)
    #                 doc = nlp((str(question)))
    #                 for token in doc: 
    #                     print("token: ", token, token.head.text, token.dep_)
    #                     print("children: ", [child for child in token.children])
    #                     dependencies[(question, token)] = (token.dep_, [child for child in token.children])

    #                 #     print(token.dep_)

    #                 # print(question)


    #             context = para['context']
    #             doc = nlp(str(context))
    #             for token in doc: 
    #                 dependencies[(context, token)] = (token.dep_, [child for child in token.children])

    # print(len(dependencies))










if __name__ == '__main__':
    # main()
    createJsonfile()
