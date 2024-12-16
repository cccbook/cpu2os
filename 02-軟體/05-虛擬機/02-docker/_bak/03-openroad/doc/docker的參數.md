
## run as a daemon

```
docker run -d --name my-container my-app:v1.0
```

如果容器已用 -d 啟動，可以使用以下方式進入 (推薦)

```
docker exec -it my-container /bin/bash
```

或者使用 docker attach

```
docker attach my-container
```

## 交談式使用

```
docker run --name my-container -it my-app:v1.0
```

