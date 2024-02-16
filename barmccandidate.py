class BARMCCandidate(object):
    __slots__ = ('indices', 'is_assigned', '_start_time', '_end_time', '_gap', '_gaps', '_totalGaps')

    def __init__(self, indices, is_assigned, start_time=0, end_time=0, gap=0, gaps=0):
        self.indices = indices
        self.is_assigned = is_assigned
        self._start_time = start_time
        self._end_time = end_time
        self._gap = gap
        self._gaps = gaps

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def gap(self):
        return self._gap
    
    @property
    def gaps(self):
        return self._gaps

    @gap.setter
    def gap(self, value):
        self._gap = value

    @gaps.setter
    def gaps(self, value):
        self._gaps = value

    @property
    def totalLength(self):
        return self._end_time - self._start_time + 1

    @property
    def totalGaps(self):
        return self._gaps / self.totalLength if self.totalLength != 0 else 0

    def __repr__(self):
        return f'<{self.__class__.__name__} {id(self)} indices={self.indices}, is_assigned={self.is_assigned}, start_time={self.start_time}, end_time={self.end_time}, gap={self.gap}, totalGaps={self.totalGaps}>'

    def __eq__(self, other):
        if isinstance(other, BAGCandidate):
            # Compare sorted tuples of indices for consistency
            return (self.start_time, self.end_time, tuple(sorted(self.indices))) == \
                   (other.start_time, other.end_time, tuple(sorted(other.indices)))
        return False

    def __hash__(self):
        # Hash a tuple of start_time, end_time, and a sorted tuple of indices
        return hash((self.start_time, self.end_time, tuple(sorted(self.indices))))
