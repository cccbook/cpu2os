# 閱讀筆記

## thread.c

yield

```cpp
thread_signal_t	thread_yield(thread_t *th)
{
	thread_t *cur_thread = thread_self();
	uint64_t now_ms;
	
	while (1) {
		if (THREAD.ready_count == 0) {
			/* there is no ready thread, host sleep for a while */
			usleep(10000); /* 10ms */
		}
		now_ms = tick_ms();

    	if ((th = pick_thread(th, now_ms)) != NULL) {
        	if (th == THREAD.active_thread)
				break; /* no other thread to yield to */

			/*
			 * switch context to DEST_THREAD 
			 * context switch may jump to stub call, so we need set 
			 * ACTIVE_THREAD here
			 */
			THREAD.active_thread = th;
        	platform_context_switch( cur_thread, th ); // 內文切換
			THREAD.active_thread = cur_thread;
			break; /* return to destination thread */
    	} else {
			/* no other thread to schedule to, so check the current active thread */
			if (THREAD.active_thread->th_state == THREAD_STATE_READY) {
				break; /* no other ready thread */
			} else {
				continue; /* check next time slot */
			}
		}
	}
	THREAD.active_thread->th_accum++;
    return THREAD.active_thread->th_signal;
}
```

create

```cpp
thread_t * thread_create(
	const char *name,
	thread_func_t func,
	void *param,
	int stacksize)
{
    thread_t *th;

    if (!THREAD.active_thread) { // initialize mapping thread management
        initial_thread_system();
    }

    if ((th=ALLOC_THREAD_SLOT())==NULL)
        return th;

    if (stacksize < 128*KB)
        stacksize = 128*KB;

	th->th_name = name;
    th->th_entry = func;
    th->th_param = param;
	th->th_alert = NULL; /* clear the alert function */
	th->th_kill_alert = NULL;
    th->th_parent = thread_self();
    th->th_signature = THREAD_SIGNATURE;
    th->th_signal = 0;
    th->th_errno = 0;
    th->th_suspcnt = 0;
    th->th_susplink = (thread_t *)NULL;

	/* create a platform dependent thread context */
	platform_create_context(th, stacksize, thread_stub);

	/* chain the thread structure */
    th->th_next = THREAD.active_thread;
    th->th_prev = THREAD.active_thread->th_prev;
    THREAD.active_thread->th_prev = th;
    th->th_prev->th_next = th;
	THREAD.ready_count ++;

    return th;
}
```

## platform.c

context-switch

```cpp
void
platform_context_switch(
	thread_t *from_th,
	thread_t *to_th
)
{
	swapcontext(&from_th->th_context.p, &to_th->th_context.p);
}
```

create context

```cpp
void
platform_create_context(
	thread_t *th,
	int stacksize,
	pfunc_t func
)
{
	memset(&th->th_context, 0, sizeof(th_context_t));
    th->th_context.th_stack = (void *)malloc( stacksize );

	assert(th->th_context.th_stack != 0);
	bzero(th->th_context.th_stack, stacksize);

	getcontext(&th->th_context.p);

	th->th_context.p.uc_link = &MAIN_THREAD->th_context.p;
	th->th_context.p.uc_stack.ss_sp = th->th_context.th_stack;
	th->th_context.p.uc_stack.ss_size = stacksize;
	makecontext(&th->th_context.p, func, 0);
}
```
