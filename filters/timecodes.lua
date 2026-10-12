--- Tie a chapter to its lecture recording.
---
--- A chapter names its lecture with a marker as the first thing in its body:
---
---     ::: {.lecture-video lecture="2"}
---     :::
---
--- (`lecture="08b"` for the discussion section). The key is resolved against
--- tools/playlist.tsv, the one place a video id is written down, and the
--- marker is replaced by a line linking the whole recording with its length.
--- From there on, `[12:34]{.t}` or `[1:02:03]{.t}` in the prose becomes a link
--- to that moment of the recording. Timecodes are checked against the video's
--- length, so a typo fails the render instead of shipping a link into the void.
---
--- Why a marker in the body and not a key in the front matter: the book PDF is
--- one document made by concatenating every chapter, and chapter front matter
--- does not survive that. A marker is positional, so the same filter reads it
--- on a single HTML page and, chapter after chapter, in the merged PDF.
---
--- On an HTML page the resolved video is also written to the `lecture-video`
--- metadata key, which filters/seo-metadata.lua reads to declare it in the
--- page's JSON-LD; this filter must therefore run before that one.

local stringify = pandoc.utils.stringify

--- Normalise a lecture key the way tools/fetch_lecture.sh does: 2 -> "02",
--- "8b" -> "08b".
local function lecture_key(raw)
  local digits, suffix = raw:match '^(%d+)(%a?)$'
  if not digits then
    return raw
  end
  return string.format('%02d', tonumber(digits)) .. suffix
end

local playlist = nil

local function lookup(key)
  if not playlist then
    playlist = {}
    local root = quarto.project and quarto.project.directory or '.'
    local path = root .. '/tools/playlist.tsv'
    local file = io.open(path, 'r')
    if not file then
      error('timecodes: cannot open ' .. path)
    end
    for line in file:lines() do
      local k, id, seconds, title = line:match '^([^\t]+)\t([^\t]+)\t(%d+)\t(.+)$'
      if k then
        playlist[k] = { id = id, seconds = tonumber(seconds), title = title }
      end
    end
    file:close()
  end
  local video = playlist[key]
  if not video then
    error('timecodes: lecture "' .. key .. '" is not in tools/playlist.tsv')
  end
  return video
end

--- "1:02:03" or "12:34" -> seconds, or nil if it is not a timecode.
local function parse(text)
  local h, m, s = text:match '^(%d+):(%d%d):(%d%d)$'
  if not h then
    m, s = text:match '^(%d+):(%d%d)$'
    h = 0
  end
  if not m or tonumber(s) > 59 or (tonumber(h) > 0 and tonumber(m) > 59) then
    return nil
  end
  return tonumber(h) * 3600 + tonumber(m) * 60 + tonumber(s)
end

local function clock(seconds)
  local h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
  if h > 0 then
    return string.format('%d:%02d:%02d', h, m, s)
  end
  return string.format('%d:%02d', m, s)
end

local function watch_url(video, seconds)
  local url = 'https://www.youtube.com/watch?v=' .. video.id
  if seconds and seconds > 0 then
    url = url .. '&t=' .. seconds .. 's'
  end
  return url
end

local function watch_line(video)
  return pandoc.Div(
    pandoc.Para({
      pandoc.Str 'Lecture video:', pandoc.Space(),
      pandoc.Link(video.title, watch_url(video)),
      pandoc.Space(), pandoc.Str '·', pandoc.Space(),
      pandoc.Str(clock(video.seconds)),
    }),
    pandoc.Attr('', { 'cs-watch' }))
end

local function timecode(span, video)
  local text = stringify(span.content)
  if not video then
    error('timecodes: [' .. text .. ']{.t} comes before any {.lecture-video} marker')
  end
  local seconds = parse(text)
  if not seconds then
    error('timecodes: "' .. text .. '" is not a timecode (want M:SS or H:MM:SS)')
  end
  if seconds > video.seconds then
    error('timecodes: ' .. text .. ' is past the end of ' .. video.title ..
      ' (' .. clock(video.seconds) .. ')')
  end
  return pandoc.Link(
    { pandoc.Str(clock(seconds)) },
    watch_url(video, seconds),
    'Watch this part of the lecture, from ' .. clock(seconds),
    pandoc.Attr('', { 'cs-t' }))
end

function Pandoc(doc)
  local current, first = nil, nil
  local blocks = pandoc.List()
  for _, block in ipairs(doc.blocks) do
    if block.t == 'Div' and block.classes:includes 'lecture-video' then
      local key = block.attributes.lecture
      if not key then
        error('timecodes: a {.lecture-video} marker without lecture="N"')
      end
      current = lookup(lecture_key(key))
      first = first or current
      blocks:insert(watch_line(current))
    else
      local video = current
      blocks:insert(block:walk {
        Span = function(span)
          if span.classes:includes 't' then
            return timecode(span, video)
          end
        end,
      })
    end
  end
  doc.blocks = blocks

  if first and FORMAT:match 'html' then
    doc.meta['lecture-video'] = {
      url = pandoc.MetaString(watch_url(first)),
      title = pandoc.MetaString('Stanford CS229, Autumn 2018 — ' .. first.title),
    }
  end
  return doc
end
