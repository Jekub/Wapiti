/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2010  CNRS
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

#include <pthread.h>

#include "model.h"
#include "tools.h"
#include "thread.h"

/******************************************************************************
 * Multi-threading code
 *
 *   This module handle the thread managment code using POSIX pthreads, on
 *   non-POSIX systems you will have to rewrite this using your systems threads.
 *   all code who depend on threads is located here so this process must not be
 *   too difficult.
 ******************************************************************************/
struct job_s {
	size_t size;
	size_t send;
	size_t batch;
	pthread_mutex_t lock;
};

typedef struct mth_s mth_t;
struct mth_s {
	job_t  *job;
	int     id;
	int     cnt;
	func_t *f;
	void   *ud;
};

/* mth_getjob:
 *   Get a new bunch of sequence to process. This function will return a new
 *   batch of sequence to process starting at position <pos> and with size
 *   <cnt> and return true. If no more batch are available, return false.
 *   This function use a lock to ensure thread safety as it will be called by
 *   the multiple workers threads.
 */
bool mth_getjob(job_t *job, size_t *cnt, size_t *pos) {
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
void mth_spawn(mdl_t *mdl, func_t *f, int W, void *ud[W]) {
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	// First prepare the jobs scheduler
	job_t job;
	job.size = mdl->train->nseq;
	job.send = 0;
	job.batch = 64;
	if (pthread_mutex_init(&job.lock, NULL) != 0)
		fatal("failed to create mutex");
	// We prepare the parameters structures that will be send to the threads
	// with informations for calling the user function.
	mth_t p[W];
	for (int w = 0; w < W; w++) {
		p[w].job = &job;
		p[w].id  = w;
		p[w].cnt = W;
		p[w].f   = f;
		p[w].ud  = ud[w];
	}
	// We are now ready to spawn the threads and wait for them to finish
	// their jobs. So we just create all the thread and try to join them
	// waiting for there return.
	pthread_t th[W];
	for (int w = 0; w < W; w++)
		if (pthread_create(&th[w], &attr, &mth_stub, &p[w]) != 0)
			fatal("failed to create thread");
	for (int w = 0; w < W; w++)
		if (pthread_join(th[w], NULL) != 0)
			fatal("failed to join thread");
	pthread_attr_destroy(&attr);
}

