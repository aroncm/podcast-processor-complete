import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { triggerEpisodeProcessing, triggerBatchProcessing, ProcessEpisodeResult } from '../lib/modalApi';
import { MODAL_CONFIG } from '../config/modal';
import {
  PlayCircle, Loader2, AlertCircle, CheckCircle, DollarSign,
  Clock, MessageSquare, Plus, Trash2, Calendar, Settings, Zap, Power, Activity,
  ChevronDown, ChevronUp
} from 'lucide-react';

interface PodcastFeed {
  id: string;
  name: string;
  rss_url: string;
  active: boolean;
  last_checked?: string;
  test_feed?: boolean;
}

interface ProcessingStats {
  totalEpisodes: number;
  totalQuotes: number;
  avgQuotesPerEpisode: number;
  totalCost: number;
}

interface ProcessedEpisode {
  episode_name: string;
  podcast_name: string;
  created_at: string;
  date_published: string;
  quote_count: number;
  duration_minutes: number;
  processing_cost: number;
}

type DatePreset = 'last7' | 'last30' | 'custom';

const EpisodeQueue: React.FC = () => {
  const [feeds, setFeeds] = useState<PodcastFeed[]>([]);
  const [stats, setStats] = useState<ProcessingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [result, setResult] = useState<ProcessEpisodeResult | null>(null);
  const [batchResult, setBatchResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Feed management
  const [showAddFeed, setShowAddFeed] = useState(false);
  const [newFeedName, setNewFeedName] = useState('');
  const [newFeedUrl, setNewFeedUrl] = useState('');
  const [addingFeed, setAddingFeed] = useState(false);

  // Date range selection
  const [datePreset, setDatePreset] = useState<DatePreset>('last7');
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');
  const [selectedFeeds, setSelectedFeeds] = useState<string[]>([]);

  // Automation settings
  const [automationEnabled, setAutomationEnabled] = useState(false);
  const [automationLogs, setAutomationLogs] = useState<any[]>([]);
  const [loadingAutomation, setLoadingAutomation] = useState(false);
  const [togglingAutomation, setTogglingAutomation] = useState(false);

  // Processed episodes
  const [processedEpisodes, setProcessedEpisodes] = useState<ProcessedEpisode[]>([]);
  const [showEpisodesList, setShowEpisodesList] = useState(false);
  const [loadingEpisodes, setLoadingEpisodes] = useState(false);

  useEffect(() => {
    loadData();
    loadAutomationSettings();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      console.log('🔄 Loading data...');

      // Load podcast feeds
      const { data: feedsData, error: feedsError } = await supabase
        .from('test_podcast_feeds')
        .select('*');

      console.log('📡 Supabase query result:', { feedsData, feedsError });

      if (feedsError) throw feedsError;

      console.log(`✅ Loaded ${feedsData?.length || 0} feeds`);
      setFeeds(feedsData || []);

      // Auto-select all active feeds
      if (feedsData) {
        const activeFeeds = feedsData.filter(f => f.active);
        console.log(`🎯 Auto-selecting ${activeFeeds.length} active feeds:`, activeFeeds.map(f => f.id));
        setSelectedFeeds(activeFeeds.map(f => f.id));
      }

      // Load processing stats
      const { data: quotesData, error: quotesError } = await supabase
        .from('test_quotes')
        .select('episode_name, processing_cost');

      if (quotesError) throw quotesError;

      if (quotesData) {
        const uniqueEpisodes = new Set(quotesData.map(q => q.episode_name));
        const totalQuotes = quotesData.length;
        const totalEpisodes = uniqueEpisodes.size;

        // Calculate total cost (sum all processing costs)
        const totalCost = quotesData.reduce((sum, q) => {
          return sum + (q.processing_cost || 0);
        }, 0);

        setStats({
          totalEpisodes,
          totalQuotes,
          avgQuotesPerEpisode: totalEpisodes > 0 ? totalQuotes / totalEpisodes : 0,
          totalCost: totalCost,
        });
      }
    } catch (err) {
      console.error('Error loading data:', err);
      setError('Failed to load podcast feeds');
    } finally {
      setLoading(false);
    }
  };

  const loadAutomationSettings = async () => {
    try {
      setLoadingAutomation(true);

      // Load automation enabled/disabled setting
      const { data: settingsData, error: settingsError } = await supabase
        .from('automation_settings')
        .select('value')
        .eq('key', 'automated_processing_enabled')
        .single();

      if (settingsError && settingsError.code !== 'PGRST116') { // Ignore not found error
        console.error('Error loading automation settings:', settingsError);
      } else if (settingsData) {
        setAutomationEnabled(settingsData.value);
      }

      // Load recent automation logs
      const { data: logsData, error: logsError } = await supabase
        .from('automation_logs')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(10);

      if (logsError && logsError.code !== 'PGRST116') {
        console.error('Error loading automation logs:', logsError);
      } else if (logsData) {
        setAutomationLogs(logsData);
      }
    } catch (err) {
      console.error('Error loading automation settings:', err);
    } finally {
      setLoadingAutomation(false);
    }
  };

  const toggleAutomation = async () => {
    try {
      setTogglingAutomation(true);
      const newValue = !automationEnabled;

      const { error } = await supabase
        .from('automation_settings')
        .update({ value: newValue, updated_at: new Date().toISOString() })
        .eq('key', 'automated_processing_enabled');

      if (error) throw error;

      setAutomationEnabled(newValue);
    } catch (err) {
      console.error('Error toggling automation:', err);
      setError('Failed to update automation settings');
    } finally {
      setTogglingAutomation(false);
    }
  };

  const loadProcessedEpisodes = async () => {
    try {
      setLoadingEpisodes(true);

      // Query to get unique episodes with aggregated data
      const { data, error } = await supabase
        .from('test_quotes')
        .select('episode_name, podcast_name, created_at, date_published, processing_cost, duration_minutes')
        .order('created_at', { ascending: false });

      if (error) throw error;

      if (data) {
        // Group by episode_name and podcast_name to get unique episodes
        const episodeMap = new Map<string, ProcessedEpisode>();

        data.forEach((quote) => {
          const key = `${quote.episode_name}|${quote.podcast_name}`;

          if (!episodeMap.has(key)) {
            episodeMap.set(key, {
              episode_name: quote.episode_name,
              podcast_name: quote.podcast_name,
              created_at: quote.created_at,
              date_published: quote.date_published,
              quote_count: 1,
              duration_minutes: quote.duration_minutes || 0,
              processing_cost: quote.processing_cost || 0,
            });
          } else {
            const existing = episodeMap.get(key)!;
            existing.quote_count += 1;
            // Use the earliest created_at
            if (new Date(quote.created_at) < new Date(existing.created_at)) {
              existing.created_at = quote.created_at;
            }
          }
        });

        // Convert map to array and sort by date
        const episodes = Array.from(episodeMap.values())
          .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
          .slice(0, 20); // Show latest 20 episodes

        setProcessedEpisodes(episodes);
      }
    } catch (err) {
      console.error('Error loading processed episodes:', err);
    } finally {
      setLoadingEpisodes(false);
    }
  };

  const toggleEpisodesList = () => {
    if (!showEpisodesList && processedEpisodes.length === 0) {
      loadProcessedEpisodes();
    }
    setShowEpisodesList(!showEpisodesList);
  };

  const handleAddFeed = async () => {
    if (!newFeedName.trim() || !newFeedUrl.trim()) {
      setError('Please enter both feed name and RSS URL');
      return;
    }

    try {
      setAddingFeed(true);
      setError(null);

      const { data, error: insertError } = await supabase
        .from('test_podcast_feeds')
        .insert({
          name: newFeedName.trim(),
          rss_url: newFeedUrl.trim(),
          active: true
        })
        .select()
        .single();

      if (insertError) throw insertError;

      setFeeds(prev => [data, ...prev]);
      setSelectedFeeds(prev => [...prev, data.id]);
      setNewFeedName('');
      setNewFeedUrl('');
      setShowAddFeed(false);
    } catch (err) {
      console.error('Error adding feed:', err);
      setError('Failed to add podcast feed');
    } finally {
      setAddingFeed(false);
    }
  };

  const handleRemoveFeed = async (feedId: string) => {
    if (!confirm('Are you sure you want to remove this feed?')) return;

    try {
      const { error: deleteError } = await supabase
        .from('test_podcast_feeds')
        .delete()
        .eq('id', feedId);

      if (deleteError) throw deleteError;

      setFeeds(prev => prev.filter(f => f.id !== feedId));
      setSelectedFeeds(prev => prev.filter(id => id !== feedId));
    } catch (err) {
      console.error('Error removing feed:', err);
      setError('Failed to remove feed');
    }
  };

  const toggleFeedActive = async (feedId: string, currentStatus: boolean) => {
    try {
      const { error } = await supabase
        .from('test_podcast_feeds')
        .update({ active: !currentStatus })
        .eq('id', feedId);

      if (error) throw error;

      setFeeds(prev =>
        prev.map(feed =>
          feed.id === feedId ? { ...feed, active: !currentStatus } : feed
        )
      );
    } catch (err) {
      console.error('Error updating feed status:', err);
      setError('Failed to update feed status');
    }
  };

  const toggleFeedSelection = (feedId: string) => {
    setSelectedFeeds(prev =>
      prev.includes(feedId)
        ? prev.filter(id => id !== feedId)
        : [...prev, feedId]
    );
  };

  const getDateRange = (): { start: Date; end: Date } => {
    const end = new Date();
    let start = new Date();

    switch (datePreset) {
      case 'last7':
        start.setDate(end.getDate() - 7);
        break;
      case 'last30':
        start.setDate(end.getDate() - 30);
        break;
      case 'custom':
        if (customStartDate && customEndDate) {
          start = new Date(customStartDate);
          end.setTime(new Date(customEndDate).getTime());
        }
        break;
    }

    return { start, end };
  };

  const handleProcessNextEpisode = async () => {
    if (selectedFeeds.length === 0) {
      setError('Please select at least one podcast feed');
      return;
    }

    try {
      setProcessing(true);
      setError(null);
      setResult(null);
      setBatchResult(null);

      const { start, end } = getDateRange();

      // Convert dates to ISO strings for the API
      const startDate = start.toISOString().split('T')[0]; // YYYY-MM-DD
      const endDate = end.toISOString().split('T')[0];   // YYYY-MM-DD

      // Process with selected feeds and date range
      const processingResult = await triggerEpisodeProcessing(
        undefined, // episode_index
        selectedFeeds, // feed_ids
        startDate, // start_date
        endDate    // end_date
      );
      setResult(processingResult);

      if (processingResult.success) {
        setTimeout(() => loadData(), 2000);
      }
    } catch (err) {
      console.error('Error processing episode:', err);
      setError('Failed to trigger episode processing');
    } finally {
      setProcessing(false);
    }
  };

  const handleBatchProcessAll = async () => {
    if (selectedFeeds.length === 0) {
      setError('Please select at least one podcast feed');
      return;
    }

    if (!confirm('This will process ALL unprocessed episodes within the date range. This may take a long time and incur significant processing costs. Continue?')) {
      return;
    }

    try {
      setBatchProcessing(true);
      setError(null);
      setResult(null);
      setBatchResult(null);

      const { start, end } = getDateRange();
      const startDate = start.toISOString().split('T')[0];
      const endDate = end.toISOString().split('T')[0];

      const result = await triggerBatchProcessing(
        selectedFeeds,
        startDate,
        endDate,
        3  // Process max 3 episodes per batch to prevent timeouts
      );

      setBatchResult(result);

      if (result.success) {
        setTimeout(() => loadData(), 3000);
      }
    } catch (err) {
      console.error('Error batch processing:', err);
      setError('Failed to trigger batch processing');
    } finally {
      setBatchProcessing(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-[#38a9d4]" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Processing Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
            <div className="flex items-center gap-3 mb-2">
              <PlayCircle className="h-5 w-5 text-[#38a9d4]" />
              <span className="text-sm text-[#b3b3b3]">Episodes Processed</span>
            </div>
            <p className="text-3xl font-bold text-white">{stats.totalEpisodes}</p>
          </div>

          <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
            <div className="flex items-center gap-3 mb-2">
              <MessageSquare className="h-5 w-5 text-[#38a9d4]" />
              <span className="text-sm text-[#b3b3b3]">Total Quotes</span>
            </div>
            <p className="text-3xl font-bold text-white">{stats.totalQuotes}</p>
          </div>

          <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
            <div className="flex items-center gap-3 mb-2">
              <Clock className="h-5 w-5 text-[#38a9d4]" />
              <span className="text-sm text-[#b3b3b3]">Avg Quotes/Episode</span>
            </div>
            <p className="text-3xl font-bold text-white">
              {stats.avgQuotesPerEpisode.toFixed(1)}
            </p>
          </div>

          <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
            <div className="flex items-center gap-3 mb-2">
              <DollarSign className="h-5 w-5 text-[#38a9d4]" />
              <span className="text-sm text-[#b3b3b3]">Total API Cost</span>
            </div>
            <p className="text-3xl font-bold text-white">
              ${stats.totalCost.toFixed(2)}
            </p>
          </div>
        </div>
      )}

      {/* Podcast Feeds Management */}
      <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Podcast Feeds</h2>
          <button
            onClick={() => setShowAddFeed(!showAddFeed)}
            className="flex items-center gap-2 px-4 py-2 bg-[#38a9d4] hover:bg-[#2c8ab0] text-white rounded-lg transition-colors"
          >
            <Plus className="h-4 w-4" />
            Add Feed
          </button>
        </div>

        {showAddFeed && (
          <div className="mb-4 p-4 bg-[#2d2d2d] rounded-lg border border-[#444]">
            <h3 className="text-white font-medium mb-3">Add New Podcast Feed</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-[#b3b3b3] mb-1">Feed Name</label>
                <input
                  type="text"
                  value={newFeedName}
                  onChange={(e) => setNewFeedName(e.target.value)}
                  placeholder="e.g., The Tim Ferriss Show"
                  className="w-full px-3 py-2 bg-[#1e1e1e] text-white border border-[#444] rounded-lg focus:outline-none focus:border-[#38a9d4]"
                  disabled={addingFeed}
                />
              </div>
              <div>
                <label className="block text-sm text-[#b3b3b3] mb-1">RSS Feed URL</label>
                <input
                  type="url"
                  value={newFeedUrl}
                  onChange={(e) => setNewFeedUrl(e.target.value)}
                  placeholder="https://feeds.example.com/podcast"
                  className="w-full px-3 py-2 bg-[#1e1e1e] text-white border border-[#444] rounded-lg focus:outline-none focus:border-[#38a9d4]"
                  disabled={addingFeed}
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleAddFeed}
                  disabled={addingFeed}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  {addingFeed ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Adding...
                    </>
                  ) : (
                    'Add Feed'
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowAddFeed(false);
                    setNewFeedName('');
                    setNewFeedUrl('');
                  }}
                  className="px-4 py-2 bg-[#444] hover:bg-[#555] text-white rounded-lg transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {feeds.length === 0 ? (
          <p className="text-[#b3b3b3] text-center py-8">
            No podcast feeds configured yet. Add one above to get started!
          </p>
        ) : (
          <div className="space-y-3">
            {feeds.map((feed) => (
              <div
                key={feed.id}
                className="flex items-center gap-4 p-4 bg-[#2d2d2d] rounded-lg border border-[#444]"
              >
                <input
                  type="checkbox"
                  checked={selectedFeeds.includes(feed.id)}
                  onChange={() => toggleFeedSelection(feed.id)}
                  className="w-5 h-5 rounded border-[#444] text-[#38a9d4] focus:ring-[#38a9d4]"
                />

                <div className="flex-1">
                  <h3 className="text-white font-medium">{feed.name}</h3>
                  <p className="text-[#b3b3b3] text-sm mt-1 truncate">{feed.rss_url}</p>
                </div>

                <button
                  onClick={() => toggleFeedActive(feed.id, feed.active)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    feed.active
                      ? 'bg-green-600 hover:bg-green-700 text-white'
                      : 'bg-[#444] hover:bg-[#555] text-[#b3b3b3]'
                  }`}
                >
                  {feed.active ? 'Active' : 'Inactive'}
                </button>

                <button
                  onClick={() => handleRemoveFeed(feed.id)}
                  className="p-2 text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded-lg transition-colors"
                  title="Remove feed"
                >
                  <Trash2 className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Episode Processing Section */}
      <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
        <h2 className="text-xl font-bold text-white mb-4">Process Episodes</h2>

        {/* Date Range Selection */}
        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm text-[#b3b3b3] mb-2">Select Time Range</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setDatePreset('last7')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  datePreset === 'last7'
                    ? 'bg-[#38a9d4] text-white'
                    : 'bg-[#2d2d2d] text-[#b3b3b3] hover:bg-[#3d3d3d]'
                }`}
              >
                <Calendar className="h-4 w-4 inline mr-2" />
                Last 7 Days
              </button>
              <button
                onClick={() => setDatePreset('last30')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  datePreset === 'last30'
                    ? 'bg-[#38a9d4] text-white'
                    : 'bg-[#2d2d2d] text-[#b3b3b3] hover:bg-[#3d3d3d]'
                }`}
              >
                <Calendar className="h-4 w-4 inline mr-2" />
                Last 30 Days
              </button>
              <button
                onClick={() => setDatePreset('custom')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  datePreset === 'custom'
                    ? 'bg-[#38a9d4] text-white'
                    : 'bg-[#2d2d2d] text-[#b3b3b3] hover:bg-[#3d3d3d]'
                }`}
              >
                <Calendar className="h-4 w-4 inline mr-2" />
                Custom Range
              </button>
            </div>
          </div>

          {datePreset === 'custom' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-[#b3b3b3] mb-1">Start Date</label>
                <input
                  type="date"
                  value={customStartDate}
                  onChange={(e) => setCustomStartDate(e.target.value)}
                  className="w-full px-3 py-2 bg-[#2d2d2d] text-white border border-[#444] rounded-lg focus:outline-none focus:border-[#38a9d4]"
                />
              </div>
              <div>
                <label className="block text-sm text-[#b3b3b3] mb-1">End Date</label>
                <input
                  type="date"
                  value={customEndDate}
                  onChange={(e) => setCustomEndDate(e.target.value)}
                  className="w-full px-3 py-2 bg-[#2d2d2d] text-white border border-[#444] rounded-lg focus:outline-none focus:border-[#38a9d4]"
                />
              </div>
            </div>
          )}
        </div>

        {/* Processing Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={handleProcessNextEpisode}
            disabled={processing || batchProcessing || selectedFeeds.length === 0}
            className="flex items-center justify-center gap-3 px-6 py-3 bg-[#38a9d4] hover:bg-[#2c8ab0] text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {processing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <PlayCircle className="h-5 w-5" />
                Process Next Episode
              </>
            )}
          </button>

          <button
            onClick={handleBatchProcessAll}
            disabled={processing || batchProcessing || selectedFeeds.length === 0}
            className="flex items-center justify-center gap-3 px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {batchProcessing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Batch Processing...
              </>
            ) : (
              <>
                <Zap className="h-5 w-5" />
                Process All Episodes
              </>
            )}
          </button>
        </div>

        {selectedFeeds.length === 0 && (
          <p className="mt-4 text-yellow-400 text-sm flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Please select at least one podcast feed above
          </p>
        )}
      </div>

      {/* Automated Processing Controls */}
      <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Settings className="h-5 w-5 text-[#38a9d4]" />
            <h2 className="text-xl font-bold text-white">Automated Processing</h2>
          </div>

          {/* Toggle Switch */}
          <button
            onClick={toggleAutomation}
            disabled={togglingAutomation || loadingAutomation}
            className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
              automationEnabled ? 'bg-green-600' : 'bg-gray-600'
            } ${togglingAutomation ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <span
              className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                automationEnabled ? 'translate-x-7' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {loadingAutomation ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-[#38a9d4]" />
          </div>
        ) : (
          <>
            {/* Status Card */}
            <div className={`p-4 rounded-lg mb-4 ${
              automationEnabled
                ? 'bg-green-900/20 border border-green-700'
                : 'bg-gray-900/20 border border-gray-700'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                <Power className={`h-5 w-5 ${automationEnabled ? 'text-green-400' : 'text-gray-400'}`} />
                <p className={`font-medium ${automationEnabled ? 'text-green-300' : 'text-gray-400'}`}>
                  {automationEnabled ? '✅ Automation Enabled' : '⏸️ Automation Paused'}
                </p>
              </div>
              <p className={`text-sm ${automationEnabled ? 'text-green-300' : 'text-gray-400'}`}>
                {automationEnabled
                  ? 'Modal Cron runs daily at midnight UTC'
                  : 'Scheduled job will skip processing until re-enabled'}
              </p>
              <p className="text-xs text-[#b3b3b3] mt-2">
                Schedule: Daily at 00:00 UTC • Configured in full_processor.py
              </p>
            </div>

            {/* Recent Activity Log */}
            {automationLogs.length > 0 && (
              <div className="mt-4">
                <div className="flex items-center gap-2 mb-3">
                  <Activity className="h-4 w-4 text-[#38a9d4]" />
                  <h3 className="text-sm font-medium text-white">Recent Automated Runs</h3>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {automationLogs.slice(0, 5).map((log) => {
                    const statusColor = log.status === 'success'
                      ? 'text-green-400 bg-green-900/20 border-green-700'
                      : log.status === 'skipped'
                      ? 'text-yellow-400 bg-yellow-900/20 border-yellow-700'
                      : 'text-red-400 bg-red-900/20 border-red-700';

                    const formatDate = (dateStr: string) => {
                      const date = new Date(dateStr);
                      return date.toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      });
                    };

                    return (
                      <div key={log.id} className={`p-3 rounded border text-xs ${statusColor}`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-medium capitalize">{log.status}</span>
                          <span className="text-[#b3b3b3]">{formatDate(log.created_at)}</span>
                        </div>
                        {log.status === 'success' && (
                          <div className="text-[#b3b3b3]">
                            {log.quotes_extracted} quotes extracted from {log.episodes_processed} episode(s)
                          </div>
                        )}
                        {log.error_message && (
                          <div className="text-[#b3b3b3] mt-1">{log.error_message}</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Processed Episodes List */}
      <div className="bg-[#1e1e1e] p-6 rounded-lg border border-[#333]">
        <button
          onClick={toggleEpisodesList}
          className="w-full flex items-center justify-between text-left"
        >
          <div className="flex items-center gap-3">
            <PlayCircle className="h-5 w-5 text-[#38a9d4]" />
            <h2 className="text-xl font-bold text-white">Processed Episodes</h2>
            {stats && (
              <span className="text-sm text-[#b3b3b3]">({stats.totalEpisodes} total)</span>
            )}
          </div>
          {showEpisodesList ? (
            <ChevronUp className="h-5 w-5 text-[#38a9d4]" />
          ) : (
            <ChevronDown className="h-5 w-5 text-[#38a9d4]" />
          )}
        </button>

        {showEpisodesList && (
          <div className="mt-4">
            {loadingEpisodes ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-[#38a9d4]" />
              </div>
            ) : processedEpisodes.length === 0 ? (
              <p className="text-[#b3b3b3] text-center py-8">
                No episodes processed yet. Start processing episodes above!
              </p>
            ) : (
              <div className="space-y-3">
                {processedEpisodes.map((episode, index) => {
                  const formatDate = (dateStr: string) => {
                    const date = new Date(dateStr);
                    return date.toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
                    });
                  };

                  return (
                    <div
                      key={`${episode.episode_name}-${index}`}
                      className="bg-[#2d2d2d] p-4 rounded-lg border border-[#444] hover:border-[#38a9d4] transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <h3 className="text-white font-medium mb-1 truncate">
                            {episode.episode_name}
                          </h3>
                          <p className="text-[#b3b3b3] text-sm mb-2">
                            {episode.podcast_name} • Published {formatDate(episode.date_published || episode.created_at)}
                          </p>
                          <div className="flex flex-wrap items-center gap-3 text-xs text-[#b3b3b3]">
                            <span className="flex items-center gap-1">
                              <MessageSquare className="h-3 w-3" />
                              {episode.quote_count} quote{episode.quote_count !== 1 ? 's' : ''}
                            </span>
                            {episode.duration_minutes > 0 && (
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {episode.duration_minutes.toFixed(1)} min
                              </span>
                            )}
                            {episode.processing_cost > 0 && (
                              <span className="flex items-center gap-1">
                                <DollarSign className="h-3 w-3" />
                                ${episode.processing_cost.toFixed(2)}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}

                {processedEpisodes.length >= 20 && (
                  <p className="text-center text-sm text-[#b3b3b3] pt-2">
                    Showing latest 20 episodes
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Single Episode Processing Result */}
      {result && (
        <div className={`p-6 rounded-lg border ${
          result.success
            ? 'bg-green-900/20 border-green-700'
            : 'bg-red-900/20 border-red-700'
        }`}>
          <div className="flex items-start gap-3">
            {result.success ? (
              <CheckCircle className="h-6 w-6 text-green-400 flex-shrink-0" />
            ) : (
              <AlertCircle className="h-6 w-6 text-red-400 flex-shrink-0" />
            )}
            <div className="flex-1">
              <h3 className={`text-lg font-bold mb-2 ${
                result.success ? 'text-green-400' : 'text-red-400'
              }`}>
                {result.success ? 'Episode Processed Successfully!' : 'Processing Failed'}
              </h3>

              {result.success && (
                <div className="space-y-2 text-sm">
                  <p className="text-white">
                    <span className="font-medium">Episode:</span> {result.episode}
                  </p>
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-green-400" />
                      <span className="text-green-300">
                        {result.quotes_extracted} quotes extracted
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-green-400" />
                      <span className="text-green-300">
                        {result.duration_minutes} minutes
                      </span>
                    </div>
                    {result.processing_cost && (
                      <div className="flex items-center gap-2">
                        <DollarSign className="h-4 w-4 text-green-400" />
                        <span className="text-green-300">
                          ${result.processing_cost.toFixed(2)} processing cost
                        </span>
                      </div>
                    )}
                  </div>

                  {result.sample_quotes && result.sample_quotes.length > 0 && (
                    <div className="mt-4">
                      <p className="text-green-300 font-medium mb-2">Sample Quotes:</p>
                      <ul className="space-y-1">
                        {result.sample_quotes.map((quote, idx) => (
                          <li key={idx} className="text-green-300 text-xs">
                            • {quote}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {result.error && (
                <p className="text-red-300">{result.error}</p>
              )}

              {result.message && (
                <p className="text-yellow-300">{result.message}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Batch Processing Result */}
      {batchResult && (
        <div className={`p-6 rounded-lg border ${
          batchResult.success
            ? 'bg-green-900/20 border-green-700'
            : 'bg-red-900/20 border-red-700'
        }`}>
          <div className="flex items-start gap-3">
            {batchResult.success ? (
              <CheckCircle className="h-6 w-6 text-green-400 flex-shrink-0" />
            ) : (
              <AlertCircle className="h-6 w-6 text-red-400 flex-shrink-0" />
            )}
            <div className="flex-1">
              <h3 className={`text-lg font-bold mb-2 ${
                batchResult.success ? 'text-green-400' : 'text-red-400'
              }`}>
                {batchResult.message || 'Batch Processing Complete'}
              </h3>

              {batchResult.success && (
                <div className="space-y-3">
                  {/* New async response format */}
                  {batchResult.message === 'Batch processing started' ? (
                    <div className="text-center p-4 bg-blue-800/30 rounded">
                      <Loader2 className="w-8 h-8 mx-auto mb-2 text-blue-400 animate-spin" />
                      <p className="text-blue-300 font-medium">Processing up to {batchResult.max_episodes} episodes in background...</p>
                      <p className="text-blue-400 text-sm mt-2">Refresh the page to see results (episodes will appear as they complete)</p>
                    </div>
                  ) : (
                    <>
                      {/* Original sync response format (for backwards compatibility) */}
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center p-3 bg-green-800/30 rounded">
                          <p className="text-2xl font-bold text-green-300">{batchResult.processed}</p>
                          <p className="text-xs text-green-400">Processed</p>
                        </div>
                        <div className="text-center p-3 bg-red-800/30 rounded">
                          <p className="text-2xl font-bold text-red-300">{batchResult.failed}</p>
                          <p className="text-xs text-red-400">Failed</p>
                        </div>
                        <div className="text-center p-3 bg-blue-800/30 rounded">
                          <p className="text-2xl font-bold text-blue-300">{batchResult.total_found}</p>
                          <p className="text-xs text-blue-400">Total Found</p>
                        </div>
                      </div>

                      {batchResult.results && batchResult.results.length > 0 && (
                        <div className="mt-4">
                          <p className="text-green-300 font-medium mb-2">Episode Results:</p>
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {batchResult.results.map((item: any, idx: number) => (
                              <div key={idx} className={`p-2 rounded text-xs ${
                                item.status === 'success'
                                  ? 'bg-green-900/30 text-green-300'
                                  : 'bg-red-900/30 text-red-300'
                              }`}>
                                <span className="font-medium">{item.episode}</span>
                                {item.status === 'success' && item.quotes && (
                                  <span className="ml-2">({item.quotes} quotes)</span>
                                )}
                                {item.error && (
                                  <span className="ml-2 text-red-400">- {item.error}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {batchResult.error && (
                <p className="text-red-300">{batchResult.error}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="text-red-300 text-sm">{error}</div>
        </div>
      )}
    </div>
  );
};

export default EpisodeQueue;
