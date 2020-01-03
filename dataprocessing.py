# coding: utf-8
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cnn_model import TCNNConfig, TextCNN


def Data_processing():
    '''
    sample 训练 测试集  = 1:1
    只含有兼职刷钻和通过
    :return:
    '''
    text = pd.read_csv('data/tribe/text_0718_0824.txt',sep = '\t',encoding='utf-8')
    text_train = text[text['dt']<=20190821]
    text_train[text_train['label_id'].isin(['通过','吸粉引流'])].groupby('label_id').count()
    text_train_tg = text_train[text_train['label_id'] == '通过'].sample(7000)
    text_train_ylxf = text_train[text_train['label_id'].isin(['吸粉引流'])]
    train = pd.concat([text_train_tg,text_train_ylxf])[['label_id','content']]
    print('train len is',len(train))
    val_tg = train[train['label_id'] == '通过'].sample(1500)
    val_ylxf = train[train['label_id'] .isin(['吸粉引流'])].sample(1500)
    val = pd.concat([val_tg,val_ylxf])
    print('val len is',len(val))
    train_1= train[~train.index.isin(val.index)]

    train_1.to_csv('data/tribe/train.txt',header=None,index=None,sep = '\t')
    val.to_csv('data/tribe/val.txt',header=None,index=None,sep = '\t')

def Data_processing_1():
    '''
    sample 训练 测试集合为自然分布 = 2w:1
    只含有兼职刷钻和通过
    :return:
    '''
    text = pd.read_csv('data/tribe/text_0718_0824.txt',sep = '\t',encoding='utf-8')
    text_0822_0824 = text[(text['dt']>=20190822) & (text['dt']<=20190824)]
    text_1 = text_0822_0824[text_0822_0824['label_id'].isin(['通过','吸粉引流'])][['label_id','content']]
    text_1.to_csv('data/tribe/text.txt',header= None,index=None,sep = '\t')

    text_train = text[text['dt']<=20190821]
    train_2 = text_train[text_train['label_id'].isin(['通过','吸粉引流'])][['label_id','content']]
    train_2.to_csv('data/tribe/train.txt',header= None,index=None,sep = '\t')


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if True:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def Data_processiing_4(catename):
    '''
    所有类别，引流吸粉 vs 其他
    比例 20：1
    :return：
    '''
    text = pd.read_csv('data/tribe/huifu_0801_0901.txt', sep='\t', encoding='utf-8').rename(columns = {'内容':'content','标注结果':'label_id'})
    text_0901_0903 = text[(text['dt'] >= 20190828) & (text['dt'] <= 20190901)]
    text_0901_0903['label_id'] = text_0901_0903['label_id'].apply(lambda x: catename if  x == catename else '其他')
    test_other = text_0901_0903[text_0901_0903['label_id']=='其他']
    test_yinliu = text_0901_0903[text_0901_0903['label_id']==catename]
    test = pd.concat([test_other.sample(len(test_yinliu)*20),test_yinliu])[['label_id','content']]
    test.to_csv('data/yinliuModel_huifu/test.txt', header=None, index=None, sep='\t')

    text_train = text[text['dt'] <= 20190827]
    text_train['label_id'] = text_train['label_id'].apply(lambda x: catename if  x == catename else '其他')
    train_other = text_train[text_train['label_id']=='其他']
    train_yinliu = text_train[text_train['label_id']==catename]
    train = pd.concat([train_other.sample(len(train_yinliu)*20),train_yinliu])[['label_id','content']]
    train.to_csv('data/yinliuModel_huifu/train.txt', header=None, index=None, sep='\t')
    return test,train

def split_data(catename):
    text = pd.read_csv('data/tribe/huifu_0801_0901.txt', sep='\t', encoding='utf-8').rename(columns = {'内容':'content','标注结果':'label_id'})
    text= text[['label_id','content']]
    text['label_id'] = text['label_id'].apply(lambda x: catename if  x == catename else '其他')
    text_tag = text[text['label_id'] == catename]
    text_qita = text[text['label_id'] == '其他']
    text_1 = pd.concat([text_tag,text_qita.sample(len(text_tag)*20)])
    trainX,testX,trainY,testY= train_test_split(text_1['content'],text_1['label_id'],test_size=0.2)
    train = pd.concat([trainY,trainX],1)
    test = pd.concat([testY,testX],1)
    train.to_csv('data/shuazuanModel_huifu/train.txt', header=None, index=None, sep='\t')
    test.to_csv('data/shuazuanModel_huifu/test.txt', header=None, index=None, sep='\t')
    return train,test

def Dataprocessing_jianzhi():
    black_sample = pd.read_excel('data/shuazuanModel/blackSample_jianzhi.xlsx').rename(columns = {'label':'label_id'})
    text = pd.read_csv('data/tribe/DATA_0801_0903.txt', sep='\t', encoding='utf-8')
    text['label_id'] = text['label_id'].apply(lambda x: '刷钻兼职' if  x == '刷钻兼职' else '其他')
    white_sample = text.sample(len(black_sample)*20)
    smaple= pd.concat([black_sample[['label_id','content','dt']],white_sample[['label_id','content','dt']]])
    smaple_test = smaple[(smaple['dt'] >= 20190901) & (smaple['dt'] <= 20190903)].drop('dt',1)
    smaple_train = smaple[smaple['dt'] <= 20190831].drop('dt',1)
    smaple_test.to_csv('data/shuazuanModel/test_clean.txt', header=None, index=None, sep='\t')
    smaple_train.to_csv('data/shuazuanModel/train_clean.txt', header=None, index=None, sep='\t')
    return smaple_test,smaple_train


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            print(line)
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels



def model_convert(model_path,pb_path):
    config = TCNNConfig()
    model = TextCNN(config)
    save_path = model_path
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver_1 = tf.train.Saver()
        saver_1.restore(sess=session, save_path=save_path)  # 读取保存的模型
        print([n.name for n in session.graph.as_graph_def().node])
        frozen_graph_def= tf.graph_util.convert_variables_to_constants(
            session,
            session.graph_def,
            output_node_names=["keep_prob","input_x","score/predict"])

        with open(pb_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

        # builder = tf.saved_model.builder.SavedModelBuilder('./model3')
        # builder.add_meta_graph_and_variables(session, ["mytag"])
        # builder.save()

if __name__ =='__main__':
    # save_path = 'checkpoints/checkpoints_white_huifu/best_validation'
    # pb_path = 'checkpoints/checkpoints_white_huifu/model_white_huifu.pb'
    # model_convert(save_path,pb_path)
    # train,test= split_data('吸粉引流')
    # smaple_test,smaple_train = Dataprocessing_jianzhi()
    train,test = split_data('刷钻兼职')