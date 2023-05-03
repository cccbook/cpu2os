# HeliOS

* https://github.com/ccc-c/HeliOS

## 參考

* https://github.com/MannyPeterson/HeliOS
* [HeliOS is a Tiny Embedded OS Designed for Arduino Boards](https://www.cnx-software.com/2020/08/14/helios-is-a-tiny-embedded-os-designed-for-arduino-boards/)

HeliOS 採用協同式多工，沒有利用中斷搶先，

* Cooperative Multitasking (Run-time Balanced)
* Event Driven Multitasking
* Task Notification/Messaging
* Timers
* Managed Memory
* Non-Preemptive (no mutexs! no deadlocks! no race conditions! no headaches!)

## 範例程式

* https://github.com/MannyPeterson/HeliOS

```c
  id = xTaskAdd("TASKBLINK", &taskBlink);

  /*
   * Call xTaskWait() to place taskBlink() into a wait
   * state by passing xTaskWait() the task id. A task
   * must be in a wait state to respond to timer events.
   */
  xTaskWait(id);

  /*
   * Set the timer interval for taskBlink() to 1,000,000 microseconds
   * (1 second). HeliOS automatically begins incrementing
   * the timer for the task once the timer interval is set.
   */
  xTaskSetTimer(id, 1000000); // 協同式多工
```

```c

void setup() {

  /*
   * Declare and initialize an int to hold the
   * task id.
   */
  int id = 0;

  /*
   * Call xHeliOSSetup() to initialize HeliOS and
   * its data structures. xHeliOSSetup() must be
   * called before any other HeliOS function call.
   */
  xHeliOSSetup();

  /*
   * Set the serial data rate and begin serial
   * communication.
   */
  Serial.begin(9600);

  /*
   * Set digital GPIO pins 2 and 3 to input only.
   */
  pinMode(2, INPUT);
  pinMode(3, INPUT);

  /*
   * Attach the interrupt service routines D2ISR()
   * and D3ISR() to the interrupts for digital GPIO
   * pins 2 and 3.
   */
  attachInterrupt(digitalPinToInterrupt(2), D2ISR, RISING);
  attachInterrupt(digitalPinToInterrupt(3), D3ISR, RISING);

  /*
   * Add the task taskSerial() to HeliOS by passing
   * xTaskAdd() the friendly name of the task as well
   * as a callback pointer to the task function.
   */
  id = xTaskAdd("TASKSERIAL", &taskSerial);

  /*
   * Call xTaskWait() to place taskSerial() into a wait
   * state by passing xTaskWait() the task id. A task
   * must be in a wait state to receive notifications.
   */
  xTaskWait(id);
}
```

## 主要程式

* https://github.com/MannyPeterson/HeliOS/blob/master/src/HeliOS.c

```c
void xHeliOSLoop() {
  int waiting = 0;
  struct Task* waitingTask[WAITINGTASKSIZE];
  struct Task* runningTask = NULL;
  struct Task* task = NULL;
  unsigned long taskStartTime = 0;
  unsigned long leastRuntime = ULONG_MAX;
  heliOSCriticalBlocking = TRUE;

  /*
   * Disable interrupts while scheduler runs.
   */
#if defined(ARDUINO_ARCH_AVR)
  noInterrupts();
#elif defined(ARDUINO_ARCH_SAM)
  noInterrupts();
#elif defined(ARDUINO_ARCH_SAMD)
  noInterrupts();
#else
        #error “HeliOS is currently supported on the Arduino AVR, SAM and SAMD architectures. Other architectures may require porting of HeliOS.”
#endif
  TaskListRewind();
  do {
    task = TaskListGet();
    if (task) {
      if (task->state == TaskStateRunning && task->totalRuntime < leastRuntime) {
        leastRuntime = task->totalRuntime;
        runningTask = task;
      } else if (task->state == TaskStateWaiting) {
        if(waiting < WAITINGTASKSIZE) {
          waitingTask[waiting] = task;
          waiting++;
        }
      }
    }
  } while (TaskListMoveNext());

  /*
   * Re-enable interrupts after sceduler runs.
   */
#if defined(ARDUINO_ARCH_AVR)
  interrupts();
#elif defined(ARDUINO_ARCH_SAM)
  interrupts();
#elif defined(ARDUINO_ARCH_SAMD)
  interrupts();
#else
        #error “HeliOS is currently supported on the Arduino AVR, SAM and SAMD architectures. Other architectures may require porting of HeliOS.”
#endif
  for(int i = 0; i < waiting; i++) {
    if (waitingTask[i]->notifyBytes > 0) {
      taskStartTime = NOW();
      (*waitingTask[i]->callback)(waitingTask[i]->id);
      waitingTask[i]->lastRuntime = NOW() - taskStartTime;
      waitingTask[i]->totalRuntime += waitingTask[i]->lastRuntime;
      waitingTask[i]->notifyBytes = 0;
    } else if (waitingTask[i]->timerInterval > 0) {
      if (NOW() - waitingTask[i]->timerStartTime > waitingTask[i]->timerInterval) {
        taskStartTime = NOW();
        (*waitingTask[i]->callback)(waitingTask[i]->id);
        waitingTask[i]->lastRuntime = NOW() - taskStartTime;
        waitingTask[i]->totalRuntime += waitingTask[i]->lastRuntime;
        waitingTask[i]->timerStartTime = NOW();
      }
    }
  }
  if(runningTask) {
    taskStartTime = NOW();
    (*runningTask->callback)(runningTask->id);
    runningTask->lastRuntime = NOW() - taskStartTime;
    runningTask->totalRuntime += runningTask->lastRuntime;
  }
  heliOSCriticalBlocking = FALSE;
}

```