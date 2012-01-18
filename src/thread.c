/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2012  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>

#include "model.h"
#include "tools.h"
#include "thread.h"
#include "wapiti.h"

/******************************************************************************
 * Multi-threading code
 *
 *   This module handle the thread managment code using POSIX pthreads, on
 *   non-POSIX systems you will have to rewrite this using your systems threads.
 *   all code who depend on threads is located here so this process must not be
 *   too difficult.
 *   If you don't want to use multithreading on non-POSIX system, just enable
 *   the definition of MTH_ANSI in wapiti.h. This will disable multithreading.
 *
 *   The jobs system is a simple scheduling system, you have to provide the
 *   number of jobs to be done and the size of each batch, a call to getjob will
 *   return the index of the first available and the size of the batch, and mark
 *   these jobs as done. This is usefull if your jobs are numbered but you can't
 *   do a trivial split as each of them may require different amount of time to
 *   be completed like gradient computation which depend on the length of the
 *   sequences.
 *   If you provide a count of 0, the job system is disabled.
 ******************************************************************************/
#ifdef MTH_ANSI
struct job_s {
	uint32_t size;
};

bool mth_getjob(job_t *job, uint32_t *cnt, uint32_t *pos) {
	if (job->size == 0)
		return false;
	*cnt = job->size;
	*pos = 0;
	job->size = 0;
	return true;
}

void mth_spawn(func_t *f, uint32_t W, void *ud[W], uint32_t size, uint32_t batch) {
	unused(batch);
	if (size == 0) {
		f(NULL, 0, 1, ud[0]);
	} else {
		job_t job = {size};
		f(&job, 0, 1, ud[0]);
	}
}

#else

#include <pthread.h>

struct job_s {
	uint32_t size;
	uint32_t send;
	uint32_t batch;
	pthread_mutex_t lock;
};

typedef struct mth_s mth_t;
struct mth_s {
	job_t    *job;
	uint32_t  id;
	uint32_t  cnt;
	func_t   *f;
	void     *ud;
};

/* mth_getjob:
 *   Get a new bunch of sequence to process. This function will return a new
 *   batch of sequence to process starting at position <pos> and with size
 *   <cnt> and return true. If no more batch are available, return false.
 *   This function use a lock to ensure thread safety as it will be called by
 *   the multiple workers threads.
 */
bool mth_getjob(job_t *job, uint32_t *cnt, uint32_t *pos) {
	if (job == NULL)
		return false;
	if (job->send == job->size)
		return false;
	pthread_mutex_lock(&job->lock);
	*cnt = min(job->batch, job->size - job->send);
	*pos = job->send;
	job->send += *cnt;
	pthread_mutex_unlock(&job->lock);
	return true;
}

static void *mth_stub(void *ud) {
	mth_t *mth = (mth_t *)ud;
	mth->f(mth->job, mth->id, mth->cnt, mth->ud);
	return NULL;
}

/* mth_spawn:
 *   This function spawn W threads for calling the 'f' function. The function
 *   will get a unique identifier between 0 and W-1 and a user data from the
 *   'ud' array.
 */
void mth_spawn(func_t *f, uint32_t W, void *ud[W], uint32_t size, uint32_t batch) {
	// First prepare the jobs scheduler
	job_t job, *pjob = NULL;
	if (size != 0) {
		pjob = &job;
		job.size = size;
		job.send = 0;
		job.batch = batch;
		if (pthread_mutex_init(&job.lock, NULL) != 0)
			fatal("failed to create mutex");
	}
	// We handle differently the case where user requested a single thread
	// for efficiency.
	if (W == 1) {
		f(&job, 0, 1, ud[0]);
		return;
	}
	// We prepare the parameters structures that will be send to the threads
	// with informations for calling the user function.
	mth_t p[W];
	for (uint32_t w = 0; w < W; w++) {
		p[w].job = pjob;
		p[w].id  = w;
		p[w].cnt = W;
		p[w].f   = f;
		p[w].ud  = ud[w];
	}
	// We are now ready to spawn the threads and wait for them to finish
	// their jobs. So we just create all the thread and try to join them
	// waiting for there return.
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_t th[W];
	for (uint32_t w = 0; w < W; w++)
		if (pthread_create(&th[w], &attr, &mth_stub, &p[w]) != 0)
			fatal("failed to create thread");
	for (uint32_t w = 0; w < W; w++)
		if (pthread_join(th[w], NULL) != 0)
			fatal("failed to join thread");
	pthread_attr_destroy(&attr);
}
#endif

