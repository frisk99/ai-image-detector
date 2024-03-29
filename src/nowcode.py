def work(n):
    path = []
    ans = []
    def dfs(i,j):
        if(i>=n):
            if i == n:
                ans.append(path[:])
            return
        for x in range(j+1,n):
            if i+x>n:
                break
            path.append(x)
            dfs(i+x,x)
            path.pop()
    dfs(0,0)
    return ans
print(work(5))
print(len(work(5)))