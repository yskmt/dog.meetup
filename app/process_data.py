import pandas as pd






sql_query = """
SELECT DISTINCT id,latitude,longitude,datetaken,description,tags,url_t,url_m
FROM photo_data_table
"""

query_results = pd.read_sql_query(sql_query, con)

# convert latlon to xy coordinate in km
xy = query_results[['latitude', 'longitude']]\
     .apply(lambda x: latlon_to_dist(x, query_latlon), axis=1)
     # .apply(lambda x: (x[0], x[1]), axis=1)
xy = pd.DataFrame(xy, columns=['xy'])   
for n, col in enumerate(['x', 'y']):
    xy[col] = xy['xy'].apply(lambda location: location[n])

query_results['x'] = xy['x']
query_results['y'] = xy['y']

# convert datetaken to hour taken
# scale: 1.0 means that 1 hour corresponds to 1 km
scale = 1.0
hours = query_results['datetaken'].apply(lambda x: x.hour+x.minute/60.0)
xyh = pd.concat([xy[['x', 'y']], hours*scale], axis=1)

query_results['hour'] = hours
    

# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
labels = DBSCAN(eps=0.3, metric='euclidean', min_samples=5,
                random_state=0)\
                .fit_predict(xyh)
                # .fit_predict(xy[['x','y']])

                        
# add labels to dataframe
query_results = pd.concat([query_results,
                           pd.DataFrame(labels, columns=['label'])],
                          axis=1)

# KDE
kde = KernelDensity(bandwidth=0.2,
                    kernel='gaussian', algorithm='ball_tree')
# kde = GMM(n_components=10, covariance_type='full')

kde.fit(query_results[['x','y','hour']])

kde_score = np.exp(kde.score_samples(query_results[['x','y','hour']]))

kde_score_max = np.sort(kde_score)[::-1][len(kde_score)/5]
kde_score /= (kde_score_max/5.0)
kde_score[kde_score>5.0] = 5.0

query_results = pd.concat(
    [query_results,
     pd.DataFrame(kde_score, index=query_results.index,
                  columns=['kde_score'])], axis=1)

# save kde model
joblib.dump(kde, 'kde.pkl')
