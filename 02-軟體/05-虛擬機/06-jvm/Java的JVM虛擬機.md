# Java 的 JVM 虛擬機

Java 雖然是一個程式語言，但是由於設計訴求上的跨平台目標，使得 Java 特別適合用虛擬機來執行，由於 Java 已經是廣為使用的程式語言，因此，其虛擬機也就變得相當重要。目前最常見的 Java 虛擬機是昇陽的 JVM 虛擬機，但是在開放原始碼界還有 Kaffe、Jikes、Novell Mono, Apache Harmony, Google Dalvik, …等數十個虛擬機，JVM 並非唯一的 Java 虛擬機。

假如使用者撰寫了一個 Java 程式 (例如：HelloWorld.java)，此時，可以利用 Java 的編譯指令 (例如：javac HelloWorld.java)，將該程式編譯成 bytecode (例如：HelloWorld.class)。接著，再利用 Java 的載入指令 (例如：java HelloWorld)，呼叫 Java 的虛擬機器 JVM，將 bytecode 載入到記憶體當中解譯並執行。若您的電腦當中安裝了 Java 的 JDK 開發工具，您可以按照範例 9.3 的方式，將一個Java 程式編譯成 bytecode，然後執行看看。

```java
class HelloWorld {
  public static void main(String[] args) {
     System.out.println("Hello World!");
  }
}
```

編譯執行

```
$ javac HelloWorld.java

$ java HelloWorld
Hello World!
```

![](../img/javaFlow.png)


Java 的 Bytecode 可以直接在網路上流通，被執行前並不需要重新編譯，JVM 虛擬機會利用 Bytecode 解譯器執行 bytecode，或利用即時編譯器 (Just in Time
Compiler : JIT) 將 Bytecode 進一步轉換成該電腦上的機器碼執行，下圖顯示了 Bytecode 的這種跨平台特性。

![](../img/javaBytecode.png)

由於虛擬機本身就是一種軟體，所以在動態載入技術的實作特別簡單，因為載入器通常就屬於虛擬機的一部分，只要利用載入器載入目的檔後，再用虛擬機執行該目的碼即可。

舉例而言，以下範例中的兩行 Java 程式會動態的載入一個由變數 name 所指定的 byte code 檔案，然後利用該檔案建立出對應的物件。這樣的技術讓 Java 程式可以在詢問使用者之後，再決定要載入哪一個類別，如此，就不需要在一開始時就載入所有的函式庫，達成動態連結與動態載入的效果。甚至可以在需要的時候，才透過網路從另一台電腦中下載目的碼並執行，這也是當初 Java Applet 的用意。

```java
Class type=ClassLoader.getSystemClassLoader().loadClass(name);
Object obj = type.newInstance();
```
但可惜的是在瀏覽器大戰中 JVM 的技術遭到敗北的命運。但是後來 Adobe Macromedia 公司的 Flash 技術，基本上也是一種瀏覽器中的虛擬機，而且成功的跨越瀏覽器執行，達成跨越網路執行的效果，Flash 可以視為 Java Applet 的復活版。

而最近的 WebAssembly 則又取代了 Flash，再次將虛擬機的觀念帶進了瀏覽器。

Java 的 javac 編譯器可以對照到 C 語言的 gcc 編譯器，bytecode 相當於虛擬機器上的目的檔，可對照到 ELF 格式的目的檔，虛擬機器 JVM 則可對照到真實的 CPU。

![](../img/javaAndC.png)

雖然昇陽沒有提供 Java 組合語言的組譯器，但是您可以在網路上找到一些，例如像是 Jon Meyer's 撰寫的 Jasmin 就是一個 java 的組譯器，這種組譯器提供了
bytecode 的組譯功能。這些工具可以幫助有興趣的讀者進一步的理解 Java 的組合語言。

在昇陽公司所提供的 Java 開發工具 JDK 中，雖然不包含 Java 的組譯器，但是包含了反組譯器 javap。

您可以利用 javap 指令將 bytecode 反組譯成組合語言，以下範顯示了 javap 反組譯器的用法，讀者可以用javap 觀察 java byte code 所對應的組合語言，以學習 Java 的組合語言寫法。

```java
$ javap -c HelloWorld.class
Compiled from "HelloWorld.java"
class HelloWorld {
  HelloWorld();
    Code:
       0: aload_0
       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
       4: return

  public static void main(java.lang.String[]);
    Code:
       0: getstatic     #7                  // Field java/lang/System.out:Ljava/io/PrintStream;
       3: ldc           #13                 // String Hello World!
       5: invokevirtual #15                 // Method java/io/PrintStream.println:(Ljava/lang/String;)V
       8: return
}
```

