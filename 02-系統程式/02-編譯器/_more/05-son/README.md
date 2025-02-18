
https://github.com/SeaOfNodes/Simple

Jserv: 

這份撰寫中的電子書展示 Sea-of-Nodes (簡稱 SoN) 編譯器中間表示法 (IR)。SoN IR 是若干具備工業強度編譯器的主體 IR，包括 HotSpot 的 C2 編譯器、Google 的 V8 編譯器，及 Oracle 的 Graal 編譯器。

作者採用一種類似 C 但功能更簡單的語言，儘管 SoN IR 的設計目的是為了在這些編譯器系統中產生機器碼，但基於展示目的，編譯器後端的實作不是該書著重的議題。

本書揭露如何以 Sea-of-Nodes IR 建構現代編譯器，而後者將資料和控制相依性合併為單一的圖 (graph)，其中每道指令都表示為一個節點 (noe)，每個相依性都表示為節點之間的邊 (edge)。不同於傳統的控制流程圖 (CFG) 及 basic block，SoN IR 使用特殊的控制節點來開始和結束程式碼區域。

SoN IR 的資料部分與 SSA (static single assignment form) 相似， 每個變數只能指派一次，而 SoN IR 不同於 SSA 之處在於，SoN IR 沒有明確的變數、版本或名稱，所有東西都是由計算節點和節點之間的邊來表示。此外，SoN IR 還包含一個特殊的 PHI 節點，代表 Phi() 函數，用於處理控制流匯合。
