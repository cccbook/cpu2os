# kubernetes

* https://kubernetes.io/


## 簡介

* https://zh.wikipedia.org/wiki/Kubernetes

Kubernetes 是一個協助我們自動化部署、擴張以及管理容器應用程式(containerized applications)的系統。相較於需要手動部署每個容器化應用程式(containers)到每台機器上，Kubernetes 可以幫我們做到以下幾件事情：

同時部署多個 containers 到一台機器上，甚至多台機器。

管理各個 container 的狀態。如果提供某個服務的 container 不小心 crash 了，Kubernetes 會偵測到並重啟這個 container，確保持續提供服務，將一台機器上所有的 containers 轉移到另外一台機器上。

* [Kubernetes 30天學習筆記](https://ithelp.ithome.com.tw/users/20103753/ironman/1590)
    * [Minikube 安裝與配置](https://ithelp.ithome.com.tw/articles/10192490)
    * Minikube 是由 Google 發布的一個輕量級工具。讓開發者可以在本機上輕易架設一個 Kubernetes Cluster，快速上手 Kubernetes 的指令與環境。Minikube 會在本機上跑起一個 virtual machine，並且在這 VM 裡建立一個 signle-node Kubernetes Cluster，本身並不支援 HA (High availability)，也不推薦在實際應用上運行。



提供機器高度擴張性。Kubernetes cluster 可以從一台機器，延展到多台機器共同運行。