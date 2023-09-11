# git

## 專案

* https://github.com/git/git

## 說明

* [Kevin Lin:facebook: Git底層的觀念](https://www.facebook.com/groups/git.tw/permalink/3675547972493844/)
    * [高見龍 - 你知道 Git 是怎麼一回事嗎](https://kaochenlong.com/2017/08/11/git-on-modern-web/)
    * [Git Internals - Git Objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)

以下轉錄自上文《Kevin Lin:facebook: Git底層的觀念》。

## 1. Blob

Comment : Data,FileContent

當我們使用 "git add [FileA]" 的時候，這會將"FileA" 加入到 ".git/index"(索引區)

"git add [FileA]" 也會使用 "SHA1" (Secure Hash Algorithm 1) 這種 hash加密法，去產生 "FileA" 內容的 hash值，這個 透過 "SHA1" (Secure Hash Algorithm 1)，加密的 "FileA" 內容 所產生 hash value，就是所謂的 "Blob" Git Object。

"Blob" Git Object 是以 "SHA1" (Secure Hash Algorithm 1) output 的格式，來儲存 "FileA" 的內容，但是 "Blob" Git Object  不  包含  "FileInformation" (檔案資訊)，比如說 "FileName"(檔案名字) 和 "FilePath"(檔案路徑)。

結論就是 "Blob" Git Object 是以 "SHA1" (Secure Hash Algorithm 1) output 的格式的 "Data"(資料) or "FileContent"(檔案內容) "Blob" Git Object 儲存在 ".git\objects".

## 2. Tree

Comment : Folder&FileInformation ; ContainBlobsOrSubTrees

當我們使用 "git commit -m [CommitMessage]"，這會 針對每一個"Folder" (包括"RootFolder")，去產生一個相對應的  "Tree" Git Object，並且  "git commit -m [CommitMessage]"，還會產生一個"Commit" Git Object。

一個 "Tree" Git Object 可以包含 多個 "Tree" Git Object 當它的 "SubTree"，這是因為 一個 "Folder" 也可以包含 多個 "Folder" 當它的 "SubFolder"。

一個 "Tree" Git Object 可以包含 多個 "Blob" Git Object ，這是因為 一個 "Folder" 也可以包含 多個 "File" 

"Tree" Git Object 儲存在 ".git\objects"

"Tree" Git Object 儲存著 "Folder&FileInformation"(資料夾&檔案資訊), 比如說 "FileName"(檔案名字) 和 "FilePath"(檔案路徑)

## 3. Commit

Comment : ContainTrees

當我們使用 "git commit -m [CommitMessage]",這會 針對每一個"Folder" (包括"RootFolder") 
去產生一個相對應的  "Tree" Git Object，並且  "git commit -m [CommitMessage]" ，還會產生一個"Commit" Git Object

"parent" 一定會指向 前一個Commit

E.g.

"1stCommit" 一定是沒有 "Parent"

"2ndCommit" 的 "Parent" 一定會是 "1stCommit"

"3rdCommit" 的 "Parent" 一定會是 "2ndCommit"

"Commit" Git Object 儲存在 ".git\objects"

"Commit" Git Object 儲存著  "CommitInformation" (Commit資訊)

"Commit" Git Object 一定會有一個"Tree" Git Object 來對應到 "RootFolder"

"Commit" Git Object 也儲存著 "Parent"(前一個commit), 

"Author"(作者), "Committer"(提交人), "Timestamp"(時間戳記), "TimeZone"(時區)

## 4. Summary - Blob, Tree, Commit

"Blob,Tree,Commit" 都是 Git Object 儲存在 ".git\objects".

