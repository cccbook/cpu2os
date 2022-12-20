# Ubuntu Java

* https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-ubuntu-18-04


```
root@localhost:/home/guest/pitifulvm# apt install default-jre
Reading package lists... Done 
Building dependency tree       
Reading state information... Done
The following package was automatically installed and is no longer required:
  libecj-java
Use 'apt autoremove' to remove it.
The following additional packages will be installed:
  at-spi2-core ca-certificates-java default-jre-headless fonts-dejavu-extra libatk-bridge2.0-0 libatk-wrapper-java
  libatk-wrapper-java-jni libatspi2.0-0 libfontenc1 libgif7 libxaw7 libxft2 libxmu6 libxxf86dga1 openjdk-11-jre
  openjdk-11-jre-headless x11-utils
Suggested packages:
  libnss-mdns fonts-ipafont-gothic fonts-ipafont-mincho fonts-wqy-microhei | fonts-wqy-zenhei fonts-indic mesa-utils
The following NEW packages will be installed:
  at-spi2-core ca-certificates-java default-jre default-jre-headless fonts-dejavu-extra libatk-bridge2.0-0
  libatk-wrapper-java libatk-wrapper-java-jni libatspi2.0-0 libfontenc1 libgif7 libxaw7 libxft2 libxmu6 libxxf86dga1
  openjdk-11-jre openjdk-11-jre-headless x11-utils
0 upgraded, 18 newly installed, 0 to remove and 369 not upgraded.
Need to get 40.7 MB of archives.
After this operation, 183 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 http://mirrors.linode.com/ubuntu bionic/main amd64 libxft2 amd64 2.3.2-1 [36.1 kB]
Get:2 http://mirrors.linode.com/ubuntu bionic/main amd64 libxxf86dga1 amd64 2:1.1.4-1 [13.7 kB]
Get:3 http://mirrors.linode.com/ubuntu bionic/main amd64 libatspi2.0-0 amd64 2.28.0-1 [59.6 kB]
Get:4 http://mirrors.linode.com/ubuntu bionic/main amd64 at-spi2-core amd64 2.28.0-1 [47.9 kB]
Get:5 http://mirrors.linode.com/ubuntu bionic-updates/main amd64 openjdk-11-jre-headless amd64 11.0.9.1+1-0ubuntu1~18.04 [38.0 MB]
Get:6 http://mirrors.linode.com/ubuntu bionic-updates/main amd64 default-jre-headless amd64 2:1.11-68ubuntu1~18.04.1 [10.9 kB]
Get:7 http://mirrors.linode.com/ubuntu bionic-updates/main amd64 ca-certificates-java all 20180516ubuntu1~18.04.1 [12.2 kB]
Get:8 http://mirrors.linode.com/ubuntu bionic-updates/main amd64 libgif7 amd64 5.1.4-2ubuntu0.1 [30.9 kB]
Processing triggers for fontconfig (2.12.6-0ubuntu2) ...
Setting up libxaw7:amd64 (2:1.0.13-1) ...
Setting up x11-utils (7.7+3build1) ...
Setting up libatk-wrapper-java (0.33.3-20ubuntu0.1) ...
Setting up libatk-wrapper-java-jni:amd64 (0.33.3-20ubuntu0.1) ...
Setting up openjdk-11-jre-headless:amd64 (11.0.9.1+1-0ubuntu1~18.04) ...
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/rmid to provide /usr/bin/rmid (rmid) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/java to provide /usr/bin/java (java) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/keytool to provide /usr/bin/keytool (keytool) in auto mode 
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jjs to provide /usr/bin/jjs (jjs) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/pack200 to provide /usr/bin/pack200 (pack200) in auto mode 
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/rmiregistry to provide /usr/bin/rmiregistry (rmiregistry) in auto mode 89%] [###########################################################################################............]   
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/unpack200 to provide /usr/bin/unpack200 (unpack200) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jfr to provide /usr/bin/jfr (jfr) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/lib/jexec to provide /usr/bin/jexec (jexec) in auto mode       
Setting up default-jre-headless (2:1.11-68ubuntu1~18.04.1) ...################################################...........] 
Setting up openjdk-11-jre:amd64 (11.0.9.1+1-0ubuntu1~18.04) ...##################################################........]   
Setting up ca-certificates-java (20180516ubuntu1~18.04.1) ...######################################################......]   
Setting up default-jre (2:1.11-68ubuntu1~18.04.1) ...###############################################################.....]   
Processing triggers for libc-bin (2.27-3ubuntu1) ...##################################################################...]   
Processing triggers for ca-certificates (20180409) ...
Updating certificates in /etc/ssl/certs...#############################################################################..]   
0 added, 0 removed; done.
Running hooks in /etc/ca-certificates/update.d...

done.
done.
root@localhost:/home/guest/pitifulvm# java --version
openjdk 11.0.9.1 2020-11-04
OpenJDK Runtime Environment (build 11.0.9.1+1-Ubuntu-0ubuntu1.18.04)
OpenJDK 64-Bit Server VM (build 11.0.9.1+1-Ubuntu-0ubuntu1.18.04, mixed mode, sharing)
root@localhost:/home/guest/pitifulvm# apt install default-jdk
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following package was automatically installed and is no longer required:
  libecj-java
Use 'apt autoremove' to remove it.
The following additional packages will be installed:
  default-jdk-headless openjdk-11-jdk openjdk-11-jdk-headless
Suggested packages:
  openjdk-11-demo openjdk-11-source visualvm
The following NEW packages will be installed:
  default-jdk default-jdk-headless openjdk-11-jdk openjdk-11-jdk-headless
Selecting previously unselected package default-jdk-headless.
Preparing to unpack .../default-jdk-headless_2%3a1.11-68ubuntu1~18.04.1_amd64.deb ...
Unpacking default-jdk-headless (2:1.11-68ubuntu1~18.04.1) ...
Selecting previously unselected package openjdk-11-jdk:amd64.
Preparing to unpack .../openjdk-11-jdk_11.0.9.1+1-0ubuntu1~18.04_amd64.deb ...
Unpacking openjdk-11-jdk:amd64 (11.0.9.1+1-0ubuntu1~18.04) ...
Selecting previously unselected package default-jdk.
Preparing to unpack .../default-jdk_2%3a1.11-68ubuntu1~18.04.1_amd64.deb ...
Unpacking default-jdk (2:1.11-68ubuntu1~18.04.1) ...
Setting up openjdk-11-jdk-headless:amd64 (11.0.9.1+1-0ubuntu1~18.04) ...
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jar to provide /usr/bin/jar (jar) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jarsigner to provide /usr/bin/jarsigner (jarsigner) in auto modeess: [ 62%] [###############################################################........................................]   
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/javac to provide /usr/bin/javac (javac) in auto mode       
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/javadoc to provide /usr/bin/javadoc (javadoc) in auto mode 
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/javap to provide /usr/bin/javap (javap) in auto mode       
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jcmd to provide /usr/bin/jcmd (jcmd) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jdb to provide /usr/bin/jdb (jdb) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jdeprscan to provide /usr/bin/jdeprscan (jdeprscan) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jdeps to provide /usr/bin/jdeps (jdeps) in auto mode       
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jimage to provide /usr/bin/jimage (jimage) in auto mode    
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jinfo to provide /usr/bin/jinfo (jinfo) in auto mode       
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jlink to provide /usr/bin/jlink (jlink) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jmap to provide /usr/bin/jmap (jmap) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jmod to provide /usr/bin/jmod (jmod) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jps to provide /usr/bin/jps (jps) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jrunscript to provide /usr/bin/jrunscript (jrunscript) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jshell to provide /usr/bin/jshell (jshell) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jstack to provide /usr/bin/jstack (jstack) in auto mode    
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jstat to provide /usr/bin/jstat (jstat) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jstatd to provide /usr/bin/jstatd (jstatd) in auto mode    
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/rmic to provide /usr/bin/rmic (rmic) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/serialver to provide /usr/bin/serialver (serialver) in auto mode
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jaotc to provide /usr/bin/jaotc (jaotc) in auto mode       
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jhsdb to provide /usr/bin/jhsdb (jhsdb) in auto mode
Setting up openjdk-11-jdk:amd64 (11.0.9.1+1-0ubuntu1~18.04) ...#######################...................................] 
update-alternatives: using /usr/lib/jvm/java-11-openjdk-amd64/bin/jconsole to provide /usr/bin/jconsole (jconsole) in auto mode
Setting up default-jdk-headless (2:1.11-68ubuntu1~18.04.1) ...##################################.........................] 
Setting up default-jdk (2:1.11-68ubuntu1~18.04.1) ...#####################################################...............] 
root@localhost:/home/guest/pitifulvm# java --version
openjdk 11.0.9.1 2020-11-04
OpenJDK Runtime Environment (build 11.0.9.1+1-Ubuntu-0ubuntu1.18.04)
OpenJDK 64-Bit Server VM (build 11.0.9.1+1-Ubuntu-0ubuntu1.18.04, mixed mode, sharing)
root@localhost:/home/guest/pitifulvm# javac --version
javac 11.0.9.1
root@localhost:/home/guest/pitifulvm# ls
jvm  jvm.c  LICENSE  Makefile  mk  README.md  tests
root@localhost:/home/guest/pitifulvm# make
make: Nothing to be done for 'all'.
root@localhost:/home/guest/pitifulvm# make clean
root@localhost:/home/guest/pitifulvm# make
  CC+LD         jvm
root@localhost:/home/guest/pitifulvm# javac tests/Factorial.java
root@localhost:/home/guest/pitifulvm# cd tests
root@localhost:/home/guest/pitifulvm/tests# java Factorial
43954714
root@localhost:/home/guest/pitifulvm# ./jvm tests/Factorial.class
43954714
