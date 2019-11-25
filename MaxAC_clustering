def MAX_AC(X,h):
    average = np.mean(X)
    var = np.var(X)*len(X)
    sum = 0
    for i in range(len(X)-h):
        sum = sum + (X[i]-average)*(X[i+h]-average)/var
    return sum


def greedyClustering(k,X,h):
    classify=[[] for i in range(k)]
    classifyIndex = [1 for i in range(k)]
    result = [-1 for i in range(len(X))]
    #取出随机的k个值
    rs = random.sample(range(0,len(X)),k)
    #把随机取出来的按index先分类
    X = np.array(X)
    for i in range(k):
        result[rs[i]] = i
        classify[i] = X[rs[i]]
    for i in range(len(X)):
        if i in rs:
            continue
        else:
            temp = X[i]
            maxDistance = [0 for a in range(k)]
            index = 0
            for j in range(k):
                tempMAX = MAX_AC((temp + classify[j])/(classifyIndex[j] + 1),h)
                MAX = MAX_AC(classify[j]/classifyIndex[j],h)
                maxDistance[j] = tempMAX - MAX
            index = maxDistance.index(max(maxDistance))
            classify[index] = classify[index] + temp
            classifyIndex[index] = classifyIndex[index] + 1
            result[i] = index
    return result;

