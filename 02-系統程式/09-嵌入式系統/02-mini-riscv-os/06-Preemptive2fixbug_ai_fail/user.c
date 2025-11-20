#include "os.h"

void user_task0(void)
{
	lib_puts("Task0: Created!\n");
	while (1) {
		lib_puts("Task0: Running...\n");
		for (int i=0; i<1000; i++) {
			lib_printf("loop: %d\n", i);
			lib_delay(1000);
		}
		lib_delay(1000);
	}
}

void user_task1(void)
{
	lib_puts("Task1: Created!\n");
	while (1) {
		lib_puts("Task1: Running...\n");
		lib_delay(1000);
	}
}

void user_init() {
	task_create(&user_task0);
	task_create(&user_task1);
}
