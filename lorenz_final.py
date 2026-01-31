import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import colorsys


def get_complementary_color(color):
	"""주어진 색상의 보색을 반환"""
	# hex 색상을 RGB로 변환
	if color.startswith("#"):
		r = int(color[1:3], 16) / 255.0
		g = int(color[3:5], 16) / 255.0
		b = int(color[5:7], 16) / 255.0
	elif color.startswith("rgb("):
		vals = color.replace("rgb(", "").replace(")", "").split(",")
		r = int(vals[0]) / 255.0
		g = int(vals[1]) / 255.0
		b = int(vals[2]) / 255.0
	else:
		return color
	
	# RGB를 HSV로 변환
	h, s, v = colorsys.rgb_to_hsv(r, g, b)
	
	# Hue를 180도 회전 (보색)
	h = (h + 0.5) % 1.0
	
	# HSV를 다시 RGB로 변환
	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	
	# RGB를 hex로 변환
	return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def get_year_series(df):
	if "dealYear" in df.columns:
		return pd.to_numeric(df["dealYear"], errors="coerce")
	if "연월" in df.columns:
		return pd.to_datetime(df["연월"], errors="coerce").dt.year
	if "rgstDate" in df.columns:
		return pd.to_datetime(df["rgstDate"], errors="coerce").dt.year
	raise ValueError("연도 컬럼을 찾을 수 없습니다: dealYear/연월/rgstDate 중 하나가 필요합니다.")


def compute_lorenz(counts_df):
	counts_df = counts_df.sort_values(["count", "구명"], ascending=[True, True], kind="mergesort")
	total = counts_df["count"].sum()
	if total == 0:
		return None

	n = len(counts_df)
	cum_gu = np.arange(1, n + 1) / n
	cum_tr = counts_df["count"].cumsum() / total
	x = np.insert(cum_gu, 0, 0)
	y = np.insert(cum_tr.to_numpy(), 0, 0)
	gini = 1 - 2 * np.trapezoid(y, x)

	positions = counts_df.copy()
	positions["x"] = cum_gu
	positions["y"] = cum_tr.to_numpy()
	return x, y, gini, positions


df = pd.read_csv("아파트실거래가2015_2025.csv", low_memory=False)
df["구명"] = df["구명"].astype(str).str.strip()
df["year"] = get_year_series(df)

years = [y for y in range(2015, 2026) if y in set(df["year"].dropna().astype(int))]
if not years:
	raise ValueError("2015~2025년 범위의 데이터가 없습니다.")

all_gu = sorted(df["구명"].dropna().unique())

# 색상 팔레트 설정 - 2025년은 2015년의 보색으로
palette = px.colors.qualitative.Light24
color_map = {}
for i, year in enumerate(years):
	if year == 2025:
		# 2015년의 보색을 2025년에 할당
		color_2015 = color_map[2015]
		color_map[2025] = get_complementary_color(color_2015)
	else:
		color_map[year] = palette[i % len(palette)]

# 단일 그래프: 연도별 로렌츠 곡선만 표시
fig = go.Figure()
top_lorenz_indices = []
lorenz_data = []  # 각 연도의 전체 데이터 저장

# Equality line
fig.add_trace(
	go.Scatter(
		x=[0, 1],
		y=[0, 1],
		mode="lines",
		line=dict(color="rgba(0,0,0,0.35)", dash="dash", width=0.8),
		showlegend=False,
		hoverinfo="skip",
		name="Equality",
	)
)

for i, year in enumerate(years):
	counts = (
		df[df["year"] == year]
		.groupby("구명")
		.size()
		.reindex(all_gu, fill_value=0)
	)
	counts_df = counts.reset_index()
	counts_df.columns = ["구명", "count"]

	result = compute_lorenz(counts_df)
	if result is None:
		continue
	x, y, gini, positions = result
	
	# 데이터 저장 (애니메이션용)
	lorenz_data.append({
		"x": x.tolist(),
		"y": y.tolist(),
		"gini": gini,
		"year": year
	})

	label = f"{year} G={gini:.3f}"
	base_color = color_map[year]
	
	# 초기 상태: 빈 trace로 시작 (애니메이션으로 채워짐)
	initial_color = base_color
	initial_width = 2.6

	top_lorenz_indices.append(len(fig.data))
	fig.add_trace(
		go.Scatter(
			x=[],  # 초기에는 빈 데이터
			y=[],
			mode="lines+text",
			name=f"{year} (G={gini:.3f})",
			legendgroup=str(year),
			line=dict(color=initial_color, width=initial_width),
			text=[],
			textposition="top left",
			textfont=dict(color=initial_color, size=11),
			hovertemplate="누적 구 비율: %{x:.2f}<br>누적 거래 비율: %{y:.2f}<extra>연도 %{fullData.name}</extra>",
		)
	)

# Plotly frames 생성
frames = []
steps_per_year = 30

for year_idx, year_data in enumerate(lorenz_data):
	full_x = year_data["x"]
	full_y = year_data["y"]
	gini = year_data["gini"]
	year = year_data["year"]
	num_points = len(full_x)
	
	# 현재 연도를 점진적으로 그리는 frames
	for step in range(1, steps_per_year + 1):
		progress = step / steps_per_year
		points_to_show = max(2, int(progress * num_points))
		
		frame_data = []
		
		# 모든 trace에 대한 데이터 구성
		for i in range(len(lorenz_data)):
			if i < year_idx:
				# 이전 연도: 전체 곡선 표시 (회색)
				prev_data = lorenz_data[i]
				prev_text = [None] * (len(prev_data["x"]) - 1) + [f"{prev_data['year']} G={prev_data['gini']:.3f}"]
				frame_data.append(go.Scatter(
					x=prev_data["x"],
					y=prev_data["y"],
					mode="lines+text",
					line=dict(color="rgba(200,200,200,0.6)", width=1.0),
					text=prev_text,
					textfont=dict(color="rgba(200,200,200,0.6)", size=11)
				))
			elif i == year_idx:
				# 현재 연도: 점진적으로 표시 (원래 색상)
				partial_x = full_x[:points_to_show]
				partial_y = full_y[:points_to_show]
				partial_text = [None] * points_to_show
				if step == steps_per_year:
					partial_text[-1] = f"{year} G={gini:.3f}"
				
				frame_data.append(go.Scatter(
					x=partial_x,
					y=partial_y,
					mode="lines+text",
					line=dict(color=color_map[years[i]], width=3.2),
					text=partial_text,
					textfont=dict(color=color_map[years[i]], size=11)
				))
			else:
				# 미래 연도: 빈 데이터
				frame_data.append(go.Scatter(
					x=[],
					y=[],
					mode="lines+text",
					text=[]
				))
		
		frames.append(go.Frame(
			data=frame_data,
			name=f"year_{year_idx}_step_{step}",
			traces=top_lorenz_indices
		))

# 마지막 frame: 모든 곡선을 원래 색상으로 복원
final_frame_data = []
for i, year_data in enumerate(lorenz_data):
	text = [None] * (len(year_data["x"]) - 1) + [f"{year_data['year']} G={year_data['gini']:.3f}"]
	final_frame_data.append(go.Scatter(
		x=year_data["x"],
		y=year_data["y"],
		mode="lines+text",
		line=dict(color=color_map[years[i]], width=2.6),
		text=text,
		textfont=dict(color=color_map[years[i]], size=11)
	))

frames.append(go.Frame(
	data=final_frame_data,
	name="final_restore",
	traces=top_lorenz_indices
))

fig.frames = frames

fig.update_layout(
	title="서울시 연도별 로렌츠 곡선 (거래 집중도)",
	width=1400,
	height=900,
	legend_title_text="연도 (클릭: 선택/해제, 더블클릭: 전체)",
	legend=dict(
		x=1.02,
		y=0.98,
		xanchor="left",
		yanchor="top",
	),
	margin=dict(t=100, l=60, r=200, b=60),
	plot_bgcolor="white",
	paper_bgcolor="white",
	xaxis=dict(
		title="누적 구 비율",
		range=[0, 1],
		showgrid=False,
		zeroline=False,
		constrain="domain",
	),
	yaxis=dict(
		title="누적 거래 비율",
		range=[0, 1],
		showgrid=False,
		zeroline=False,
		scaleanchor="x",
		scaleratio=1,
	),
)

output_path = "lorenz_seoul_interactive.html"
fig.write_html(output_path, include_plotlyjs="cdn")

# ============================================================================
# Trajectory Plot: 각 구의 연도별 위치 이동 시각화
# ============================================================================

# 각 연도별 구의 위치 데이터 수집
trajectory_data = {}  # {구명: [(year, x, y), ...]}

for year in years:
	counts = (
		df[df["year"] == year]
		.groupby("구명")
		.size()
		.reindex(all_gu, fill_value=0)
	)
	counts_df = counts.reset_index()
	counts_df.columns = ["구명", "count"]
	
	result = compute_lorenz(counts_df)
	if result is None:
		continue
	x, y, gini, positions = result
	
	# 각 구의 위치 저장
	for idx, row in positions.iterrows():
		gu_name = row["구명"]
		if gu_name not in trajectory_data:
			trajectory_data[gu_name] = []
		trajectory_data[gu_name].append((year, row["x"], row["y"]))

# Trajectory plot 생성 - 초기에는 빈 상태
fig_trajectory = go.Figure()

# Equality line만 추가
fig_trajectory.add_trace(
	go.Scatter(
		x=[0, 1],
		y=[0, 1],
		mode="lines",
		line=dict(color="rgba(0,0,0,0.35)", dash="dash", width=0.8),
		showlegend=False,
		hoverinfo="skip",
		name="Equality",
	)
)

# 구별 색상 팔레트
gu_colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
gu_color_map = {gu: gu_colors[i % len(gu_colors)] for i, gu in enumerate(sorted(all_gu))}

# 초기 안내 문구 annotation
fig_trajectory.add_annotation(
	x=0.5,
	y=0.5,
	text="<b>👈 우측 패널에서 구를 선택하세요</b><br>선택된 구의 궤적만 표시됩니다",
	showarrow=False,
	font=dict(size=16, color="rgba(100,100,100,0.6)"),
	xref="x",
	yref="y",
	name="guide_text"
)

# 프레임은 JavaScript에서 동적으로 생성
fig_trajectory.frames = []

# 슬라이더 및 버튼 추가
sliders = [dict(
	active=0,
	yanchor="top",
	y=-0.15,
	xanchor="left",
	currentvalue=dict(
		prefix="연도: ",
		visible=True,
		xanchor="right"
	),
	pad=dict(b=10, t=50),
	len=0.9,
	x=0.1,
	steps=[]  # JavaScript에서 동적 생성
)]

updatemenus = [dict(
	type="buttons",
	showactive=False,
	x=0.05,
	y=-0.15,
	xanchor="left",
	yanchor="top",
	buttons=[
		dict(
			label="▶ 재생",
			method="skip",  # JavaScript에서 처리
			args=[]
		),
		dict(
			label="⏸ 일시정지",
			method="skip",
			args=[]
		)
	]
)]

fig_trajectory.update_layout(
	title=f"서울시 구별 Trajectory (로렌츠 곡선 상 위치 이동) - {years[-1]}년",
	width=1400,
	height=900,
	xaxis=dict(
		title="누적 구 비율",
		range=[0, 1],
		showgrid=True,
		gridcolor="rgba(0,0,0,0.1)",
		zeroline=False,
		constrain="domain",
	),
	yaxis=dict(
		title="누적 거래 비율",
		range=[0, 1],
		showgrid=True,
		gridcolor="rgba(0,0,0,0.1)",
		zeroline=False,
		scaleanchor="x",
		scaleratio=1,
	),
	legend=dict(
		x=1.02,
		y=0.98,
		xanchor="left",
		yanchor="top",
		font=dict(size=10),
		itemsizing="constant",
		title="구 선택 (멀티 셀렉트)",
	),
	plot_bgcolor="white",
	paper_bgcolor="white",
	hovermode="closest",
	sliders=sliders,
	updatemenus=updatemenus,
	margin=dict(t=80, l=60, r=250, b=150),
)

trajectory_path = "lorenz_trajectory_interactive.html"
fig_trajectory.write_html(trajectory_path, include_plotlyjs="cdn")

# Trajectory plot 커스터마이징: 멀티 셀렉트 기능
with open(trajectory_path, "r", encoding="utf-8") as f:
	traj_html = f.read()

gu_list_js = json.dumps(sorted(all_gu))
gu_colors_js = json.dumps({gu: gu_color_map[gu] for gu in sorted(all_gu)})
years_list_js = json.dumps(years)
trajectory_data_js = json.dumps({
	gu: [(t[0], t[1], t[2]) for t in sorted(trajectory_data[gu], key=lambda x: x[0])]
	for gu in sorted(all_gu) if gu in trajectory_data
})

trajectory_script = f'''
<style>
.color-swatch {{
	display: inline-block;
	width: 12px;
	height: 12px;
	margin-right: 5px;
	border-radius: 2px;
	vertical-align: middle;
}}
#gu-selector {{
	position: fixed;
	right: 20px;
	top: 100px;
	background: white;
	padding: 15px;
	border: 1px solid #ccc;
	border-radius: 5px;
	max-height: 600px;
	overflow-y: auto;
	font-family: Arial, sans-serif;
	font-size: 12px;
	z-index: 1000;
	box-shadow: 0 2px 10px rgba(0,0,0,0.1);
	width: 200px;
}}
#gu-selector h4 {{
	margin: 0 0 10px 0;
	font-size: 14px;
	color: #333;
}}
.gu-item {{
	padding: 5px;
	cursor: pointer;
	border-radius: 3px;
	margin-bottom: 3px;
	transition: background 0.2s;
}}
.gu-item:hover {{
	background: #f0f0f0;
}}
.gu-item.selected {{
	background: #e3f2fd;
	font-weight: bold;
}}
#select-controls {{
	margin-bottom: 10px;
	padding-bottom: 10px;
	border-bottom: 1px solid #ddd;
}}
#select-controls button {{
	font-size: 11px;
	padding: 4px 8px;
	margin-right: 5px;
	cursor: pointer;
	border: 1px solid #ccc;
	background: white;
	border-radius: 3px;
}}
#select-controls button:hover {{
	background: #f0f0f0;
}}
</style>
<div id="gu-selector">
	<h4>🎯 구 선택</h4>
	<div id="select-controls">
		<button id="select-all">전체선택</button>
		<button id="deselect-all">전체해제</button>
	</div>
	<div id="gu-list"></div>
</div>
<script>
(function() {{
	const guList = {gu_list_js};
	const guColors = {gu_colors_js};
	const yearsList = {years_list_js};
	const trajectoryData = {trajectory_data_js};
	
	const activeWidth = 5.0;
	const activeMarkerSize = 12;
	
	let selectedGu = new Set();
	let isAnimating = false;
	let currentYearIndex = 0;
	let animationInterval = null;
	
	function getPlotDiv() {{
		return document.querySelector('.plotly-graph-div');
	}}
	
	function rebuildPlot() {{
		const plotDiv = getPlotDiv();
		if (!plotDiv) return;
		
		// 안내 문구 제거
		const annotations = plotDiv.layout.annotations || [];
		const newAnnotations = annotations.filter(a => a.name !== 'guide_text');
		
		// Equality line은 유지하고, 나머지는 제거 후 선택된 구만 추가
		const newData = [plotDiv.data[0]];  // Equality line
		
		// 선택된 구들만 trace 추가
		Array.from(selectedGu).forEach((guName, idx) => {{
			if (!trajectoryData[guName]) return;
			
			const traj = trajectoryData[guName];
			const years = traj.map(t => t[0]);
			const xData = traj.map(t => t[1]);
			const yData = traj.map(t => t[2]);
			
			newData.push({{
				x: xData.slice(0, 1),  // 초기에는 첫 점만
				y: yData.slice(0, 1),
				mode: 'lines+markers',
				name: guName,
				line: {{ color: guColors[guName], width: activeWidth }},
				marker: {{ 
					size: activeMarkerSize, 
					color: guColors[guName],
					line: {{ width: 1, color: 'white' }}
				}},
				hovertemplate: `<b>${{guName}}</b><br>연도: %{{customdata[0]}}<br>누적 구 비율: %{{x:.3f}}<br>누적 거래 비율: %{{y:.3f}}<extra></extra>`,
				customdata: [[years[0]]],
				showlegend: false
			}});
		}});
		
		// 현재 연도의 annotation 추가
		if (selectedGu.size > 0) {{
			const labelAnnotations = createLabelAnnotations(currentYearIndex);
			newAnnotations.push(...labelAnnotations);
		}}
		
		Plotly.react(plotDiv, newData, {{
			...plotDiv.layout,
			annotations: newAnnotations,
			title: selectedGu.size > 0 
				? `서울시 구별 Trajectory - ${{yearsList[currentYearIndex]}}년 (선택: ${{selectedGu.size}}개 구)`
				: '서울시 구별 Trajectory - 우측에서 구를 선택하세요'
		}});
		
		// 슬라이더 업데이트
		updateSlider();
		
		// 첫 프레임으로 리셋
		currentYearIndex = 0;
	}}
	
	function createLabelAnnotations(yearIdx) {{
		const annotations = [];
		const positions = [];
		
		Array.from(selectedGu).forEach((guName, idx) => {{
			if (!trajectoryData[guName]) return;
			
			const traj = trajectoryData[guName];
			if (yearIdx >= traj.length) return;
			
			const x = traj[yearIdx][1];
			const y = traj[yearIdx][2];
			
			// 겹침 방지를 위한 오프셋 계산
			let xOffset = 0.03;
			let yOffset = 0.03;
			
			// 이미 있는 라벨들과 너무 가까우면 조정
			positions.forEach(pos => {{
				const dist = Math.sqrt(Math.pow(pos.x - x, 2) + Math.pow(pos.y - y, 2));
				if (dist < 0.08) {{
					yOffset += 0.03;
					xOffset += 0.01;
				}}
			}});
			
			positions.push({{ x, y }});
			
			// 화면 밖으로 나가지 않도록 조정
			let finalX = x + xOffset;
			let finalY = y + yOffset;
			
			if (finalX > 0.95) finalX = x - xOffset;
			if (finalY > 0.95) finalY = y - 0.05;
			if (finalY < 0.05) finalY = 0.05;
			
			annotations.push({{
				x: finalX,
				y: finalY,
				xref: 'x',
				yref: 'y',
				text: guName,
				showarrow: true,
				arrowhead: 2,
				arrowsize: 1,
				arrowwidth: 1.5,
				arrowcolor: guColors[guName],
				ax: 0,
				ay: -20,
				bgcolor: guColors[guName],
				bordercolor: guColors[guName],
				borderwidth: 2,
				borderpad: 4,
				font: {{
					color: 'white',
					size: 11,
					family: 'Arial, sans-serif'
				}},
				opacity: 0.95
			}});
		}});
		
		return annotations;
	}}
	
	function updateFrame(yearIdx) {{
		const plotDiv = getPlotDiv();
		if (!plotDiv || selectedGu.size === 0) return;
		
		currentYearIndex = yearIdx;
		
		const updateData = [];
		
		Array.from(selectedGu).forEach((guName, idx) => {{
			if (!trajectoryData[guName]) return;
			
			const traj = trajectoryData[guName];
			const pointsToShow = Math.min(yearIdx + 1, traj.length);
			
			const years = traj.slice(0, pointsToShow).map(t => t[0]);
			const xData = traj.slice(0, pointsToShow).map(t => t[1]);
			const yData = traj.slice(0, pointsToShow).map(t => t[2]);
			
			updateData.push({{
				x: [xData],
				y: [yData],
				customdata: [years.map(y => [y])]
			}});
		}});
		
		// trace 인덱스 (Equality line 다음부터)
		const traceIndices = Array.from({{length: selectedGu.size}}, (_, i) => i + 1);
		
		Plotly.restyle(plotDiv, {{
			x: updateData.map(d => d.x[0]),
			y: updateData.map(d => d.y[0]),
			customdata: updateData.map(d => d.customdata[0])
		}}, traceIndices).then(() => {{
			// annotation 업데이트
			const newAnnotations = createLabelAnnotations(yearIdx);
			Plotly.relayout(plotDiv, {{
				annotations: newAnnotations,
				title: `서울시 구별 Trajectory - ${{yearsList[yearIdx]}}년 (선택: ${{selectedGu.size}}개 구)`
			}});
		}});
	}}
	
	function playAnimation() {{
		if (isAnimating) return;
		if (selectedGu.size === 0) {{
			alert('구를 먼저 선택해주세요!');
			return;
		}}
		
		isAnimating = true;
		
		animationInterval = setInterval(() => {{
			currentYearIndex++;
			
			if (currentYearIndex >= yearsList.length) {{
				stopAnimation();
				return;
			}}
			
			updateFrame(currentYearIndex);
			updateSliderPosition(currentYearIndex);
		}}, 500);
	}}
	
	function stopAnimation() {{
		isAnimating = false;
		if (animationInterval) {{
			clearInterval(animationInterval);
			animationInterval = null;
		}}
	}}
	
	function updateSlider() {{
		const plotDiv = getPlotDiv();
		if (!plotDiv) return;
		
		const steps = yearsList.map((year, idx) => ({{
			label: String(year),
			method: 'skip',
			args: []
		}}));
		
		Plotly.relayout(plotDiv, {{
			'sliders[0].steps': steps,
			'sliders[0].active': currentYearIndex
		}});
	}}
	
	function updateSliderPosition(idx) {{
		const plotDiv = getPlotDiv();
		if (!plotDiv) return;
		
		Plotly.relayout(plotDiv, {{
			'sliders[0].active': idx
		}});
	}}
	
	function toggleGuSelection(guName) {{
		if (selectedGu.has(guName)) {{
			selectedGu.delete(guName);
		}} else {{
			selectedGu.add(guName);
		}}
		updateGuSelector();
		currentYearIndex = 0;  // 리셋
		rebuildPlot();
	}}
	
	function updateGuSelector() {{
		const guListDiv = document.getElementById('gu-list');
		if (!guListDiv) return;
		
		guListDiv.innerHTML = '';
		guList.forEach(gu => {{
			const item = document.createElement('div');
			item.className = 'gu-item' + (selectedGu.has(gu) ? ' selected' : '');
			
			if (selectedGu.has(gu)) {{
				const swatch = document.createElement('span');
				swatch.className = 'color-swatch';
				swatch.style.backgroundColor = guColors[gu];
				item.appendChild(swatch);
			}}
			
			const text = document.createTextNode(gu);
			item.appendChild(text);
			
			item.addEventListener('click', function() {{
				toggleGuSelection(gu);
			}});
			
			guListDiv.appendChild(item);
		}});
	}}
	
	window.addEventListener('load', function() {{
		setTimeout(function() {{
			const plotDiv = getPlotDiv();
			if (!plotDiv) {{
				console.error('Plot div not found');
				return;
			}}
			
			console.log('Trajectory plot loaded');
			
			// 초기 구 선택기 렌더링
			updateGuSelector();
			
			// 전체 선택/해제 버튼
			document.getElementById('select-all').addEventListener('click', function() {{
				selectedGu = new Set(guList);
				updateGuSelector();
				currentYearIndex = 0;
				rebuildPlot();
			}});
			
			document.getElementById('deselect-all').addEventListener('click', function() {{
				selectedGu.clear();
				updateGuSelector();
				currentYearIndex = 0;
				rebuildPlot();
			}});
			
			// 재생/일시정지 버튼
			const buttons = plotDiv.querySelectorAll('.updatemenu-button');
			if (buttons.length >= 2) {{
				buttons[0].addEventListener('click', function(e) {{
					e.preventDefault();
					e.stopPropagation();
					playAnimation();
				}});
				
				buttons[1].addEventListener('click', function(e) {{
					e.preventDefault();
					e.stopPropagation();
					stopAnimation();
				}});
			}}
			
			// 슬라이더 변경 감지
			plotDiv.on('plotly_sliderchange', function(event) {{
				stopAnimation();
				const newYearIdx = event.slider.active;
				updateFrame(newYearIdx);
			}});
			
		}}, 1000);
	}});
}})();
</script>
'''

traj_html = traj_html.replace('</body>', trajectory_script + '</body>')

with open(trajectory_path, "w", encoding="utf-8") as f:
	f.write(traj_html)

print(f"Trajectory 시각화 완료: {trajectory_path}")

# ============================================================================
# 기존 로렌츠 곡선 시각화
# ============================================================================

# Legend 클릭 이벤트 커스터마이징
with open(output_path, "r", encoding="utf-8") as f:
	html_content = f.read()

years_js = json.dumps(years)
year_colors_js = json.dumps([color_map[year] for year in years])
top_lorenz_idx_js = json.dumps(top_lorenz_indices)

# 프레임 이름 목록을 미리 생성하여 순서 보장
frame_names = []
for year_idx, year_data in enumerate(lorenz_data):
	for step in range(1, steps_per_year + 1):
		frame_names.append(f"year_{year_idx}_step_{step}")
frame_names.append("final_restore")
frame_names_js = json.dumps(frame_names)

custom_script = f'''
<script>
(function() {{
	const years = {years_js};
	const yearColors = {year_colors_js};
	const topLorenzIdx = {top_lorenz_idx_js};
	const frameNames = {frame_names_js};
	const grayColor = 'rgba(200,200,200,0.6)';
	const grayWidth = 1.0;
	const activeWidth = 3.2;

	let selectedIndices = new Set(years.map((_, i) => i));
	let animationComplete = false;

	function getPlotDiv() {{
		return document.querySelector('.plotly-graph-div');
	}}

	function updateColors() {{
		if (!animationComplete) return;
		
		const plotDiv = getPlotDiv();
		if (!plotDiv) return;
		
		const lineColors = [];
		const lineWidths = [];
		const textColors = [];
		
		topLorenzIdx.forEach((idx, i) => {{
			if (selectedIndices.has(i)) {{
				lineColors.push(yearColors[i]);
				lineWidths.push(activeWidth);
				textColors.push(yearColors[i]);
			}} else {{
				lineColors.push(grayColor);
				lineWidths.push(grayWidth);
				textColors.push(grayColor);
			}}
		}});
		
		Plotly.restyle(plotDiv, {{
			'line.color': lineColors,
			'line.width': lineWidths,
			'textfont.color': textColors
		}}, topLorenzIdx);
	}}

	// 애니메이션 자동 재생
	function playAnimation() {{
		const plotDiv = getPlotDiv();
		if (!plotDiv || !plotDiv.data) {{
			console.error('Plot not ready');
			return;
		}}
		
		console.log('Starting animation with', frameNames.length, 'frames');
		
		// frameNames에서 final_restore를 제외한 나머지 프레임들
		const animationFrames = frameNames.slice(0, -1);
		
		// 애니메이션 실행
		Plotly.animate(plotDiv, animationFrames, {{
			frame: {{ duration: 50, redraw: true }},
			transition: {{ duration: 0 }},
			mode: 'immediate'
		}}).then(() => {{
			console.log('Animation completed');
			// 애니메이션 완료 후 3초 대기
			setTimeout(() => {{
				// final_restore frame으로 이동
				Plotly.animate(plotDiv, ['final_restore'], {{
					frame: {{ duration: 0, redraw: true }},
					transition: {{ duration: 500 }},
					mode: 'immediate'
				}}).then(() => {{
					console.log('Final restore completed');
					// 인터랙션 활성화
					animationComplete = true;
				}}).catch(err => {{
					console.error('Final restore error:', err);
					animationComplete = true;
				}});
			}}, 3000);
		}}).catch(err => {{
			console.error('Animation error:', err);
			animationComplete = true;
		}});
	}}

	// Legend 클릭 이벤트 처리
	if (typeof Plotly === 'undefined') {{
		console.error('Plotly is not loaded!');
		return;
	}}
	
	console.log('Plotly loaded, version:', Plotly.version);
	
	window.addEventListener('load', function() {{
		console.log('Window loaded');
		
		setTimeout(function() {{
			const plotDiv = getPlotDiv();
			if (!plotDiv) {{
				console.error('Plot div not found');
				return;
			}}
			
			console.log('Plot div found:', plotDiv.id);
			console.log('Plot data length:', plotDiv.data ? plotDiv.data.length : 'no data');
			console.log('Frame names:', frameNames.length, 'frames');
			
			plotDiv.on('plotly_legendclick', function(data) {{
				if (!animationComplete) return false;
				
				const curveNumber = data.curveNumber;
				const yearIndex = topLorenzIdx.indexOf(curveNumber);
				
				if (yearIndex !== -1) {{
					if (selectedIndices.has(yearIndex)) {{
						selectedIndices.delete(yearIndex);
					}} else {{
						selectedIndices.add(yearIndex);
					}}
					updateColors();
				}}
				
				return false;
			}});
			
			plotDiv.on('plotly_legenddoubleclick', function(data) {{
				if (!animationComplete) return false;
				
				if (selectedIndices.size === years.length) {{
					selectedIndices.clear();
				}} else {{
					selectedIndices = new Set(years.map((_, i) => i));
				}}
				updateColors();
				return false;
			}});
			
			// 애니메이션 시작
			console.log('Starting animation...');
			playAnimation();
		}}, 1000);
	}});
}})();
</script>
'''

html_content = html_content.replace('</body>', custom_script + '</body>')

with open(output_path, "w", encoding="utf-8") as f:
	f.write(html_content)

print(f"시각화 완료: {output_path}")
print(f"파일 경로: file://{output_path}")
# fig.show()  # 서버 문제로 주석 처리
