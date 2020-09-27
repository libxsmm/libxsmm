
from torch.autograd.profiler import *

class FunctionEventAvgNested(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    def __init__(self):
        self.key = None
        self.count = 0
        self.node_id = 0
        self.is_async = False
        self.is_remote = False
        self.cpu_time_total = 0
        self.cuda_time_total = 0
        self.self_cpu_time_total = 0
        self.input_shapes = None
        self.cpu_memory_usage = 0
        self.cuda_memory_usage = 0
        self.self_cpu_memory_usage = 0
        self.self_cuda_memory_usage = 0

    def add(self, other):
        if self.key is None:
            self.key = other.nested_key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
        assert isinstance(other, FunctionEvent)
        assert other.nested_key == self.key
        self.cpu_time_total += other.cpu_time
        self.cuda_time_total += other.cuda_time
        self.self_cpu_time_total += other.self_cpu_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.cuda_memory_usage += other.cuda_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_cuda_memory_usage += other.self_cuda_memory_usage
        self.count += other.count
        #self.count += 1
        return self

    def __iadd__(self, other):
        return self.add(other)

    def __repr__(self):
        return '<FunctionEventAvg cpu_time={} cuda_time={} key={}>'.format(
            self.cpu_time_str, self.cuda_time_str, self.key)


def el_nested_key_averages(self, only_top_level):
    """Averages all function events over their keys.

    Returns:
        An EventList containing FunctionEventAvg objects.
    """
    self.populate_cpu_children()
    stats = defaultdict(FunctionEventAvgNested)
    for evt in self:
        if only_top_level and getattr(evt, "parent", None): continue
        stats[evt.nested_key] += evt
    if only_top_level:
        for evt in stats.values():
            evt.self_cpu_time_total = evt.cpu_time_total
        # for evt in self:
        #     if not evt.cuda_time: continue
        #     parent = getattr(evt, "parent", None)
        #     if parent:
        #         pp = getattr(parent, "parent", None)
        #         while pp:
        #             parent = pp
        #             pp = getattr(parent, "parent", None)
        #         stats[parent.nested_key].cuda_time_total += evt.cuda_time

    return EventList(stats.values())

def p_nested_key_averages(self, only_top_level=False):
    self._check_finish()
    return self.function_events.nested_key_averages(only_top_level)

def fe_append_cpu_child(self, child):
    """Append a CPU child of type FunctionEvent.

    One is supposed to append only dirrect children to the event to have
    correct self cpu time being reported.
    """
    assert(isinstance(child, FunctionEvent))
    self.cpu_children.append(child)
    child.parent = self

def fe_nested_key(self):
    plist = [self.name]
    nested_name = getattr(self, "nested_name", None)
    if nested_name:
        return nested_name
    p = getattr(self, "parent", None)
    while p:
        plist.insert(0, p.name)
        p = getattr(p, "parent", None)
    nested_name = ".".join(plist)
    self.nested_name = nested_name
    return nested_name

def format_time(time_us):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return '{:.3f}s'.format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return '{:.3f}ms'.format(time_us / US_IN_MS)
    return '{:.3f}us'.format(time_us)

def print_op_timings(prof, use_gpu=False, prefix='prof'):
    sorted_fe = sorted(prof.function_events, key=lambda event: [event.cpu_interval.start, -event.cpu_interval.end],)
    start_time = sorted_fe[0].cpu_interval.start if len(sorted_fe) > 0 else 0
    with open("%s.OpTimings.txt" % prefix, "w") as f:
        for i, fe in enumerate(sorted_fe):
            fe_name = getattr(fe, 'nested_key', fe.name)
            cstr = ""
            if use_gpu:
               for kinfo in fe.kernels: cstr += " %10.3f %10.3f %8.3f %8.3f " % ((kinfo.interval.start - start_time) /1000.0, (kinfo.interval.end - start_time)/1000.0, (kinfo.interval.start - fe.cpu_interval.start) /1000.0, kinfo.interval.elapsed_us()/1000.0)
            print("%-6d %6d %12.4f %12.4f %12.4f %2s %s   %-40s    %s" % (i, fe.id, (fe.cpu_interval.start - start_time)/1000.0, (fe.cpu_interval.end - start_time)/1000.0, fe.cpu_interval.elapsed_us()/1000.0, fe.thread, cstr, fe_name.replace(' ', '_'), fe.input_shapes), file=f)

FunctionEvent.append_cpu_child = fe_append_cpu_child
FunctionEvent.nested_key = property(fe_nested_key)
EventList.nested_key_averages = el_nested_key_averages
profile.nested_key_averages = p_nested_key_averages
profile.nested_key_averages.__doc__ = EventList.nested_key_averages.__doc__
profile.print_op_timings = print_op_timings
