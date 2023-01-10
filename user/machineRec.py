import os
from random import uniform
from matplotlib import pyplot as plt
from yellowbrick.cluster import InterclusterDistance, KElbowVisualizer, SilhouetteVisualizer
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from random import sample
import numpy as np
import pandas as pd

def pca_trans(data, features):
    n_components = len(features)
    pca = PCA(n_components).fit(data)
    PCs = []
    for l in range(1, n_components + 1):
        PCs.append("PC" + str(l))

    # eigenvectors = np.round(pca.components_.transpose(), decimals=3)
    # pd.DataFrame(eigenvectors, index=features, columns=PCs)
    # loadings = eigenvectors * np.sqrt(eigenvalues)
    # pd.DataFrame(np.round(loadings, decimals=4), index=features, columns=PCs)
    # fig = plt.figure(figsize=(8, 5))
    # x_axis = np.arange(n_components) + 1

    # plt.plot(x_axis, eigenvalues, 'ro-', linewidth=2)
    # plt.title('Scree Plot')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    # plt.show()

    no_pc = 6
    PC_scores = pca.fit_transform(data)  # PC scores for downstream analytics
    scores = pd.DataFrame(PC_scores[:, 0:no_pc], columns=PCs[0:no_pc])
    return scores, pca

#测试数据集是否适合聚类
def hopkins_statistic(data: pd.DataFrame, sampling_ratio: float = 0.3):
    sampling_ratio = min(max(sampling_ratio, 0.1), 0.5)
    n_samples = int(data.shape[0] * sampling_ratio)
    sample_data = data.sample(n_samples)
    data = data.drop(index=sample_data.index)  # ,inplace = True)
    data_dist = cdist(data, sample_data).min(axis=0).sum()
    ags_data = pd.DataFrame({col: np.random.uniform(data[col].min(), data[col].max(), n_samples) \
                             for col in data})
    ags_dist = cdist(data, ags_data).min(axis=0).sum()
    H_value = ags_dist / (data_dist + ags_dist)
    return H_value

#获取最优KNN的K值
def get_k(scores):
    n_clust = 10
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, n_clust))

    visualizer.fit(scores)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

#获取最大最小价格日期用于做标准化
def get_minmax(df):
    max_price = max(df['Price'].tolist())
    min_price = min(df['Price'].tolist())
    max_date = max(df['Date'].tolist())
    min_date = min(df['Date'].tolist())
    return max_price, min_price, max_date, min_date

#将游戏名转化为序号
def get_gameno(u_games,df):
    game_nos = []
    for each in u_games:
        game_nos.append(df[df.Name==each].index.tolist()[0])
    return game_nos

#将用户的游戏库转化为标准的特征向量组
def get_u_vector(u_games, pca,df,data):
    global u_times
    temp = []
    for each in u_games:
        if each in df['Name'].tolist():
            temp.append(each)
    game_nos = get_gameno(temp,df)
    gameprices = []
    gamedates = []
    gamelabels = np.zeros(25)
    for each in game_nos:
        game = data.loc[each].tolist()
        gamelabel = np.array(game[2:])
        gameprices.append(game[0])
        gamedates.append(game[1])
        gamelabels += gamelabel

    gameprices = np.array(gameprices)
    gamedates = np.array(gamedates)

    mean_price = np.mean(gameprices)
    mean_date = np.mean(gamedates)

    min_label = min(gamelabels)
    max_label = max(gamelabels)
    normalized_labels = []
    for each in gamelabels:
        normalized_label = (each - min_label)/(max_label - min_label)
        normalized_labels.append(normalized_label)

    u_vector = [mean_price, mean_date]
    u_vector.extend(normalized_labels)
    print("U vector here:")
    print(u_vector)
    u_vector = pca.transform([u_vector])[:,0:6]
    return u_vector[0]

#预测5个最近游戏
def prediction(u_vector,scores):
    # cluster = kmeans.predict(u_vector, sample_weight=None)
    # for each in scores.index:
    #     if scores.loc[each, 'Cluster'] != cluster:
    #         scores = scores.drop(each)
    # scores.to_excel('cluster_2.xlsx')
    # scores = scores.drop('Cluster', axis=1)
    model_knn = NearestNeighbors(n_neighbors=3, algorithm='brute')
    model_knn.fit(scores)
    distances, indices = model_knn.kneighbors(u_vector)
    return(indices[0])

print(os.getcwd())

# df = pd.read_csv("./static/File/game_preprocessed.csv", index_col=[0])
# u_times = pd.read_csv('./static/File/users.csv', index_col=[0])
#
# game_names = df['Name'].tolist()
#
# data = df.drop(['Name','Rate'], axis=1)
# features = data.columns.tolist()
# data = MinMaxScaler().fit_transform(data)
# data = pd.DataFrame(data)
#
# scores, pca = pca_trans(data)




# k=6
# kmeans = KMeans(k, init="k-means++", n_init=10, max_iter=1000, random_state=43)
# kmeans.fit(scores)
# goodness = silhouette_score(scores, kmeans.labels_)
# print("No Clusters =", k, " Silhouette = ", goodness)
# scores['Cluster'] = kmeans.labels_

# goodness = silhouette_score(scores, clustering.labels_)
# print("No Clusters =", , " Silhouette = ", goodness)

# max_price, min_price, max_date, min_date = get_minmax(df)

#测试单个游戏的最近5个游戏
########################################
# game_no = 69
# test = data.loc[game_no].tolist()
# test = np.array(test)
# # u_games = ['Company of Heroes 2', 'Northgard', 'Dota 2', "Don't Starve Together", 'Dread Hunger', 'Age of Empires IV', 'PUBG: BATTLEGROUNDS', 'RimWorld', 'Hell Let Loose', 'Gunfire Reborn', 'The Binding of Isaac: Rebirth', 'Gremlins, Inc.', 'Terraria', 'Outward', 'Prison Architect', 'Risk of Rain 2', 'Kenshi', 'Mount & Blade II: Bannerlord', 'Valheim', 'Plebby Quest: The Crusades', 'Darkest Dungeon', 'Chrono Ark', 'Age of Empires III: Definitive Edition', 'Detroit: Become Human', 'PlateUp!', 'Crown Trick', 'Grounded', 'Kingdom Come: Deliverance', 'Thronebreaker: The Witcher Tales', 'Borderlands 3', 'Total War: THREE KINGDOMS', 'Totally Accurate Battle Simulator', 'HELLDIVERS', 'Hades', 'Eville', "Tales of Maj'Eyal", 'Stardew Valley', 'It Takes Two', 'Ring of Pain', 'Starbound', 'Divinity: Original Sin 2', 'Human: Fall Flat', 'Pit People', 'Oxygen Not Included', 'Raft', 'ELDEN RING', 'Scrap Mechanic', 'Streets of Rogue', 'Hand of Fate 2', 'Battle Brothers', 'Slay the Spire', 'Griftlands', 'Loop Hero', 'Core Keeper', 'Battlerite', "Hero's Hour", 'Urtuk: The Desolation', 'Inscryption', 'Green Hell', 'Castle Crashers', 'Root', 'Foretales', 'Necesse', 'Spiritfarer: Farewell Edition', 'Mutant Year Zero: Road to Eden', 'Fantasy of Expedition', 'Factorio', 'Mirror', 'tModLoader', 'Stacklands', 'Lovers in a Dangerous Spacetime', 'Domina', 'Bloons TD 6', 'Vagante', 'Dota Underlords', 'Dead Cells', 'Heroes of Hammerwatch', "No Man's Sky", 'Rise to Ruins', 'Hidden Through Time', 'Circle Empires', 'Just King', 'Nine Parchments', 'One Step From Eden', 'BattleBlock Theater', '球球少女', 'Cultist Simulator', 'Brothers - A Tale of Two Sons', 'Cuphead', 'Dicey Dungeons', 'Overcooked! 2', 'Sea of Thieves', 'Seek Girl:Fog Ⅰ', 'Ultimate Chicken Horse', "Yokai's Secret", 'Torchlight II', 'The Forest', 'Crypt of the NecroDancer', 'Magicka 2', 'Super Auto Pets', 'Clone Drone in the Danger Zone', 'Airships: Conquer the Skies', 'Vampire Survivors', 'Enter the Gungeon', 'Warhammer: Vermintide 2', 'This War of Mine', 'Dream Date', 'Deep Rock Galactic']
# u_games = ['Company of Heroes 2']
# ########################################
#
# u_vector = get_u_vector(u_games, pca,df,data)
# indices = prediction([u_vector])
# for each in indices:
    # print(df.loc[each, ['Name', 'Price', 'Date', 'Rate']])

def time_scaler(u_times):
    print(u_times)
    for user in u_times.index.tolist():
        times = np.array(np.nan_to_num(u_times.loc[user].tolist()))
        scaler = MinMaxScaler()
        scaler.fit(times.reshape(-1, 1))
        for game in u_times.columns.tolist():
            print("this is type")
            print(type(u_times.loc[user,game]))
            print("this is value")
            print(pd.isnull(u_times.loc[user,game]))
            isNan = pd.isnull(u_times.loc[user,game])
            # isNan = isNan.values.tolist()[0]
            print(isNan)
            # print(type(u_times.loc[game]))
            # if not pd.isnull(u_times.loc[user,game]) :
            if not isNan :
                print("we are in")
                print(u_times.loc[user, game])
                u_times.loc[user, game] = scaler.transform([[u_times.loc[user, game]]])[0][0]

# def time_scaler(u_times):
#     for user in u_times.index.tolist():
#         times = np.array(np.nan_to_num(u_times.loc[user].tolist()))
#         temp = MinMaxScaler().fit_transform(times.reshape(-1, 1)).reshape(1, -1)
#         u_times.loc[user] = temp[0]

def time_dropper(u_times,game_names):
    temp = []
    for each in u_times.columns.tolist():
        if each not in game_names:
            temp.append(each)
    u_times_dropped = u_times.drop(temp, axis=1)
    return u_times_dropped

def get_u_data(u_times,pca,df,data):
    df_u_vector = pd.DataFrame(columns=range(0,6), index=u_times.index.tolist())
    for user in u_times.index.tolist():
        temp = []
        for game in u_times.columns.tolist():
            the_time = u_times.loc[user, game]
            if not np.isnan(the_time):
                temp.append(game)
        u_vector_temp = get_u_vector(temp, pca,df,data)
        df_u_vector.loc[user] = u_vector_temp
    return df_u_vector

def collaborative_filtering(u_data, u_times,u_vector,df):
    u_names = u_data.index.tolist()
    # time_df = pd.DataFrame(index=u_names, columns=features)
    cosine_sims = cosine_similarity([u_vector], u_data)[0]
    # print(cosine_sims)
    print("Here is the u_times")
    u_times['cosine_sim'] = cosine_sims
    cal_scores = []

    for game in u_times.columns.tolist()[:-1]:
        sum_sim = 0
        times = np.nan_to_num(u_times.loc[:, game].tolist())
        gameno = get_gameno([game],df)[0]
        rate = df.loc[gameno, 'Rate'].tolist()
        for u_name in u_names:
            if not np.isnan(u_times.loc[u_name, game]):
                sum_sim += u_times.loc[u_name, 'cosine_sim']

        # print(times , '+' ,cosine_sims, '+', sum(np.multiply(cosine_sims, times)), '+', rate, "+", sum_sim)
        if sum_sim != 0:
            temp_score = (rate * sum(np.multiply(cosine_sims, times)))/sum_sim
        else:
            temp_score = (rate * sum(np.multiply(cosine_sims, times)))
        cal_scores.append(temp_score)
    u_newtimes = u_times.drop('cosine_sim', axis=1)
    u_newtimes.loc['scores'] = cal_scores
    u_finaltimes_df = u_newtimes.sort_values(by='scores', axis=1, ascending=False)
    u_finaltimes = get_gameno(u_finaltimes_df.columns.tolist()[0:2], df)
    print('u_finaltimes', u_finaltimes)
    return u_finaltimes

# u_times = time_dropper(u_times,game_names)
# time_scaler(u_times)
# u_data = get_u_data(u_times,pca,df,data)
# print(collaborative_filtering(u_data, u_times,u_vector,df))