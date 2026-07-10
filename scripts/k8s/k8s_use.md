- delete all pods

```sh
sudo kubectl -n anime-role-detect delete pods --all
```

- get pods with wide output
```sh
sudo kubectl -n anime-role-detect get pods -o wide
```

- get pod log
```sh
sudo kubectl -n anime-role-detect logs <pod-name>
```

- get svc

```sh
kubectl -n anime-role-detect get svc
```