/*
 
Copyright (c) 2011, Willem-Hendrik Thiart
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * The names of its contributors may not be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL WILLEM-HENDRIK THIART BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <assert.h>

#include "heap.h"

#define DEBUG 0
#define INITIAL_CAPACITY 13

/*-------------------------------------------------------------- INIT METHODS */

static int __child_left(const int idx)
{
    return idx * 2 + 1;
}

static int __child_right(const int idx)
{
    return idx * 2 + 2;
}

static int __parent(const int idx)
{
assert(idx != 0);
return (idx - 1) / 2;
}


/**
 * Init a heap and return it.
 * We malloc space for it.
 *
 * @param cmp : a function pointer used to determine the priority of the item
 * @param udata : udata passed through to compare callback */
heap_t *heap_new(int (*cmp) (const void *,
			     const void *,
			     const void *udata), const void *udata)
{
    heap_t *hp;

    hp = malloc(sizeof(heap_t));
    hp->cmp = cmp;
    hp->udata = udata;
    hp->arraySize = INITIAL_CAPACITY;
    hp->array = malloc(sizeof(void *) * hp->arraySize);
    hp->count = 0;
    return hp;
}

/**
 * Free memory held by heap */
void heap_free(heap_t * hp)
{
    free(hp->array);
    free(hp);
}

static void __ensurecapacity(heap_t * hp)
{
    int ii;

    void **array_n;

    if (hp->count < hp->arraySize)
	return;

    /* double capacity */
    hp->arraySize *= 2;
    array_n = malloc(sizeof(void *) * hp->arraySize);

    /* copy old data across to new array */
    for (ii = 0; ii < hp->count; ii++)
    {
	array_n[ii] = hp->array[ii];
	assert(array_n[ii]);
    }

    /* swap arrays */
    free(hp->array);
    hp->array = array_n;
}

/*------------------------------------------------------------ IN/OUT METHODS */

static void __swap(heap_t * hp, const int i1, const int i2)
{
    void *tmp = hp->array[i1];

    hp->array[i1] = hp->array[i2];
    hp->array[i2] = tmp;
}

static int __pushup(heap_t * hp, int idx)
{
    while (1)
    {
	int parent, compare;

	/* we are now the root node */
	if (0 == idx)
	    return idx;

	parent = __parent(idx);
	compare = hp->cmp(hp->array[idx], hp->array[parent], hp->udata);

	/* we are smaller than the parent */
	if (compare < 0)
	{
	    return -1;
	}
	else
	{
	    __swap(hp, idx, parent);
	}

	idx = parent;
    }

    return idx;
}

static void __pushdown(heap_t * hp, int idx)
{
    while (1)
    {
	int childl, childr, child, compare;

	childl = __child_left(idx);
	childr = __child_right(idx);
	child = -1;

	assert(idx != hp->count);

	if (childr >= hp->count)
	{
	    /* can't pushdown any further */
	    if (childl >= hp->count)
		return;

	    child = childl;
	}
	else
	{
	    /* find biggest child */
	    compare =
		hp->cmp(hp->array[childl], hp->array[childr], hp->udata);

	    if (compare < 0)
	    {
		child = childr;
	    }
	    else
	    {
		child = childl;
	    }
	}

	assert(child != -1);

	compare = hp->cmp(hp->array[idx], hp->array[child], hp->udata);

	/* idx is smaller than child */
	if (compare < 0)
	{
	    assert(hp->array[idx]);
	    assert(hp->array[child]);
	    __swap(hp, idx, child);
	    idx = child;
	    /* bigger than the biggest child, we stop, we win */
	}
	else
	{
	    return;
	}
    }
}

/**
 * Add this value to the heap.
 * @param item : the item to be added to the heap */
void heap_offer(heap_t * hp, void *item)
{
    assert(hp);
    assert(item);
    if (!item)
	return;

    __ensurecapacity(hp);

    hp->array[hp->count] = item;

    /* ensure heap properties */
    __pushup(hp, hp->count);

    hp->count++;
}

#if DEBUG
static void DEBUG_check_validity(heap_t * hp)
{
    int ii;

    for (ii = 0; ii < hp->count; ii++)
	assert(hp->array[ii]);
}
#endif

/**
 * Remove the top value from this heap.
 * @return top item of the heap */
void *heap_poll(heap_t * hp)
{
    void *item;

    assert(hp);

    if (!hp)
	return NULL;

    if (0 == heap_count(hp))
	return NULL;

#if DEBUG
    DEBUG_check_validity(hp);
#endif

    item = hp->array[0];

    hp->array[0] = NULL;
    __swap(hp, 0, hp->count - 1);
    hp->count--;

#if DEBUG
    DEBUG_check_validity(hp);
#endif

    if (hp->count > 0)
    {
	assert(hp->array[0]);
	__pushdown(hp, 0);
    }

#if DEBUG
    DEBUG_check_validity(hp);
#endif

    return item;
}

/**
 * @return the item on the top of the heap */
void *heap_peek(heap_t * hp)
{
    if (!hp)
	return NULL;

    if (0 == heap_count(hp))
	return NULL;

    return hp->array[0];
}

/**
 * Clear all items from the heap */
void heap_clear(heap_t * hp)
{
    hp->count = 0;
}

static int __item_get_idx(heap_t * hp, const void *item)
{
    int compare, idx;

    for (idx = 0; idx < hp->count; idx++)
    {
	compare = hp->cmp(hp->array[idx], item, hp->udata);

	/* we have found it */
	if (compare == 0)
	{
	    return idx;
	}
    }

    return -1;
}

/**
 * The heap will remove this item
 * @return item to be removed */
void *heap_remove_item(heap_t * hp, const void *item)
{
    void *ret_item;
    int idx;

    /* find the index */
    idx = __item_get_idx(hp, item);

    /* we didn't find it */
    if (idx == -1)
	return NULL;

    /* swap the item we found with the last item on the heap */
    ret_item = hp->array[idx];
    hp->array[idx] = hp->array[hp->count - 1];
    hp->array[hp->count - 1] = NULL;

    /* decrement counter */
    hp->count -= 1;

    /* ensure heap property */
    __pushup(hp, idx);

    return ret_item;
}

/**
 * The heap will remove this item
 * @return 1 if the heap contains this item, 0 otherwise */
int heap_contains_item(heap_t * hp, const void *item)
{
    int idx;

    /* find the index */
    idx = __item_get_idx(hp, item);

    return (idx != -1);
}

/*------------------------------------------------------------ STATUS METHODS */

/**
 * How many items are there in this heap?
 * @return number of items in heap */
int heap_count(heap_t * hp)
{
    return hp->count;
}

/*--------------------------------------------------------------79-characters-*/
