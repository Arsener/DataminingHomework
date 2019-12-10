import pandas as pd
import os


# 归一化
def normalized(data, df, col):
    max_number = df[col].max()
    min_number = df[col].min()

    data[col] = df[col].map(lambda x: float(x - min_number) / float(max_number - min_number))


# 替换
def replace(data, df, col, d):
    data[col] = df[col].map(lambda x: d[x])


# one-hot
def one_hot(data, df, col):
    one_hot_pd = pd.get_dummies(df[col])
    one_hot_pd.columns = [col + str(i) for i in range(len(set(df[col])))]
    data = pd.concat([data, one_hot_pd], axis=1)
    return data


def main():
    df = pd.read_csv(os.path.join('data', 'train_set.csv'))
    data = pd.DataFrame()

    # 保留ID（客户唯一标识）
    data['ID'] = df['ID']
    # 对age进行归一化（客户年龄）
    normalized(data, df, 'age')
    # 对job进行one-hot编码（客户的职业）
    data = one_hot(data, df, 'job')
    # 将marital替换为数字（婚姻状况）
    marital_dict = {'divorced': -1, 'single': 0, 'married': 1}
    replace(data, df, 'marital', marital_dict)
    # 将education替换为数字（受教育水平）
    education_dict = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    replace(data, df, 'education', education_dict)
    # 将default替换为数字（是否有违约记录）
    default_dict = {'no': 0, 'yes': 1}
    replace(data, df, 'default', default_dict)
    # 对balance进行归一化（每年账户的平均余额）
    normalized(data, df, 'balance')
    # 将housing替换为数字（是否有住房贷款）
    housing_dict = {'no': 0, 'yes': 1}
    replace(data, df, 'housing', housing_dict)
    # 将loan替换为数字（是否有个人贷款）
    loan_dict = {'no': 0, 'yes': 1}
    replace(data, df, 'loan', loan_dict)
    # 将contact替换为数字（与客户联系的沟通方式）
    contact_dict = {'unknown': -1, 'cellular': 0, 'telephone': 1}
    replace(data, df, 'contact', contact_dict)
    # 对day进行归一化（最后一次联系的时间（几号））
    normalized(data, df, 'day')
    # 将month替换为数字（最后一次联系的时间（月份））并归一化
    month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    replace(data, df, 'month', month_dict)
    normalized(data, data, 'month')
    # 对duration进行归一化（最后一次联系的交流时长）
    normalized(data, df, 'duration')
    # 对campaign进行归一化（在本次活动中，与该客户交流过的次数）
    normalized(data, df, 'campaign')
    # 对pdays进行归一化（距离上次活动最后一次联系该客户，过去了多久（-1表示没有联系过））
    normalized(data, df, 'pdays')
    # 对previous进行归一化（在本次活动之前，与该客户交流过的次数）
    normalized(data, df, 'previous')
    # 对poutcome进行one-hot编码（上一次活动的结果）
    data = one_hot(data, df, 'poutcome')
    # 保留标签（客户是否会订购定期存款业务）
    data['y'] = df['y']

    data.to_csv(os.path.join('data', 'processed_data.csv'), index=False)


if __name__ == '__main__':
    main()
