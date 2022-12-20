# 黃光 OPC EDA

* 搜尋用 -- https://sci-hub.se/

* [Machine Learning for Electronic Design Automation: A Survey](https://nicsefc.ee.tsinghua.edu.cn/media/publications/2020/arxiv_None_slide.pdf) (讚！)
    * https://github.com/thu-nics/awesome_ai4eda
    * https://nicsefc.ee.tsinghua.edu.cn/
    * https://nicsefc.ee.tsinghua.edu.cn/projects/eda/
    * 影片 -- https://www.bilibili.com/video/av669446228/ (非 EDA)

* [Electronic Design Automation for IC System Design](http://www.ime.cas.cn/icac/learning/learning_3/201908/P020190826566810494541.pdf)

* [Lithography Hotspot Detection: From Shallow to Deep Learning](http://www.cse.cuhk.edu.hk/~byu/papers/C62-SOCC2017-DNN.pdf) -- CNN Model
* [Adoption of OPC and the Impact on Design and Layout (PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.7001&rep=rep1&type=pdf)

* [基於影像對光學鄰近修正(OPC)後之佈局間隔檢測之前置過濾方法](https://ir.nctu.edu.tw/bitstream/11536/45952/1/560501.pdf), 研究生：林涵恩

* [關於黃光及其100個疑問，這篇文章已全面解答](https://kknews.cc/digital/onnol66.html)

* [轉錄-[心得] 半導體黃光製程工作內容分享 Vol.1 - Vol. 3](http://jinraylee.blogspot.com/2012/10/vol1-vol-3.html)

* [請問OPC光罩到底是什麼啊](https://tw.answers.yahoo.com/question/index?qid=20060901000011KK11154)

Optical Proximity Correction....光學修正...

在教學觀念上...metal line要怎麼依光罩的設計留在wafer上呢???

首先layout先寫到mask上的...經由上光阻..曝光顯影...

蝕刻..去光阻之後...在wafer的表面可以留下由光罩定義的pattern...

實際上卻不是那麼一回事...pattern在轉彎處..也就是有角度的地方...

以及pattern的最末端(line-end)...pattern的形狀很容易走樣....甚至消失...

原本是設計是完美九十度轉彎...曝完之後..會變圓弧....也就是所謂的rounding...

(就像contact和via..在layout中都是四邊形..但曝完光後...都變成圓形)

這主要是光在穿過光罩時會有一大堆光學效應...進而削弱pattern的profile...

所以...在寫光罩前..會用OPC去調整補強 layout...把可能會被削弱的pattern..

給予適當的補強...OPC跑完會產生一個GDS..這個GDS才能拿來寫光罩的...

這個GDS你可以用Laker或是Virtuoso或是其他的軟體打開來看看..

你會發覺pattern的edge變的很亂...會外突或內縮....

opc前的pattern的edge都是很smooth..跑完opc...所有的edge都變形...

歪七扭八...很難形容,,,建議你親自看看....目前能執行這個任務至少有三種..

明導的calibre...新思的hercules/proteus...OPC也分rule base和model opc...

rule base opc是指pattern的width,spacing,甚至周邊其他的pattern的width..

這三種數據形成一個OPC table...pattern落在某個區間..edge就會長出去或內縮....

calibre和hercules是rule-base opc軟體...

model opc則是用GDS中任一個區塊中所有的pattern為基準來做OPC...

rule base OPC算是較簡易型...model OPC就較複雜...

根據黃光機的不同又可以分成scanner和stepper的OPC...