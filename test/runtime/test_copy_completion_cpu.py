# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Exercise result lifetimes with the unavailable device stream API simulated."""

import gc
import weakref
from contextlib import nullcontext

import pytest

from tokenspeed.runtime.execution.copy_completion import CopyStreamPool


class Stream:
    def __init__(self):
        self.completed = False

    def wait_stream(self, other):
        pass

    def query(self):
        return self.completed

    def synchronize(self):
        self.completed = True


class Device:
    def Stream(self, *, device):
        return Stream()

    def device(self, device):
        return nullcontext()

    def stream(self, stream):
        return nullcontext()


class Source:
    pass


def test_each_pending_result_owns_its_copy_stream_until_completion():
    pool = CopyStreamPool(Device(), "unavailable-device")
    first = pool.copy_after(Stream(), ())
    with first:
        pass
    second = pool.copy_after(Stream(), ())
    assert first.stream is not second.stream
    with second:
        pass
    first.synchronize()
    third = pool.copy_after(Stream(), ())
    assert third.stream is first.stream
    with third:
        pass
    third.synchronize()
    second.synchronize()


def test_discarded_result_keeps_sources_alive_until_copy_finishes():
    pool = CopyStreamPool(Device(), "unavailable-device")
    source = Source()
    reference = weakref.ref(source)
    completion = pool.copy_after(Stream(), (source,))
    with completion:
        pass
    stream = completion.stream
    del completion, source
    gc.collect()
    assert reference() is not None
    stream.completed = True
    following = pool.copy_after(Stream(), ())
    assert reference() is None
    with following:
        pass
    following.synchronize()


def test_failed_copy_scope_releases_only_after_synchronization():
    pool = CopyStreamPool(Device(), "unavailable-device")
    completion = pool.copy_after(Stream(), ())
    with pytest.raises(RuntimeError, match="finished copy scope"):
        completion.synchronize()
    with pytest.raises(ValueError):
        with completion:
            raise ValueError("failed copy")
    assert completion.stream.completed
    following = pool.copy_after(Stream(), ())
    assert following.stream is completion.stream
    with following:
        pass
    following.synchronize()
