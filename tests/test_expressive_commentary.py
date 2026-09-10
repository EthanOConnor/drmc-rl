from tools.align_expressive_commentary import commentary_windows, consistent_prefix


def test_commentary_preserves_unresolved_video_identity():
    text = ('## Setup\nSource: `Text/apr-2026/yt_mzox-OAETaU.txt`, 00:24:07–00:24:45.\n'
            '## Other\nSource: `Text/jul-2026/day2.txt`, 00:19:00–00:19:23.\n')
    windows = list(commentary_windows(text))
    assert windows[0]['video_id'] == 'mzox-OAETaU'
    assert windows[0]['start'] == 1447
    assert windows[1]['video_id'] is None


def test_video_prefix_stops_at_gap_and_cannot_reset_from_bondless_checkpoint():
    initial = ['........']*15+['RRR....B']
    after = ['........']*15+['RRR..bbB']
    event = dict(i=0,p='BB',x=5,y=15,o=0,t=10,b=after)
    game = dict(b0=initial,events=[event,{**event,'i':2}])
    prefix, reason = consistent_prefix(game)
    assert len(prefix) == 1 and reason == 'sequence_gap'
    prefix, reason = consistent_prefix(dict(b0=after,events=[event]))
    assert prefix == [] and reason == 'initial_board_missing_or_bonds_unknown'


def test_missing_video_timestamp_is_not_invented():
    initial = ['........']*15+['RRR....B']
    event = dict(i=0,p='BB',x=5,y=15,o=0,t=None,b=['........']*15+['RRR..bbB'])
    prefix, reason = consistent_prefix(dict(b0=initial,events=[event]))
    assert prefix == [] and reason == 'timestamp_missing'
