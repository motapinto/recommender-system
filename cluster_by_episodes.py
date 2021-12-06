

def cluster_by_episodes(df):
    
    def _make_cluster(x):
        clusters=[3, 10, 30, 100]
        for i, j in enumerate (clusters):
            if x<j: return i
        return len(clusters)
    
    c=df.columns
    df=df.groupby(c[0]).count()
    df[c[1]]=df[c[1]].apply(_make_cluster)
    df[c[2]]=df[c[2]].apply(lambda x:1)
    return df
