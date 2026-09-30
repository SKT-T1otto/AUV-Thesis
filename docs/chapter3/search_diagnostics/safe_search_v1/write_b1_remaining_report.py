"""Build reproducible Markdown tables from the verified six-version analysis."""
from pathlib import Path
from chapter3_bser.experiments.safe_search_v1.run_paired import read_json, file_hash, write_json

HERE=Path(__file__).resolve().parent
OUT=HERE/'b1_remaining_results_20260929'
OPTIONS={'V0':'原始 B1','V1':'A','V2':'A+B','V3':'C','V4':'A+B+C','V5':'A+C'}


def pct(value):
    return f'{value*100:.2f}%'


def paired(value):
    lo,hi=value['bootstrap_95_percentile_ci']
    return f"{value['mean_delta']*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]"


def write():
    a=read_json(OUT/'analysis.json');c=read_json(OUT/'cases.json')['cases']
    gains=read_json(OUT/'gain_cases.json')
    assert a['complete'] and len(c)==120 and read_json(OUT/'numeric_audit.json')['passed']
    assert gains['complete']
    lines=['# B1 V1 / V2 / V4 补充实验与六版本配对结果','',
      f"新增 60 个真实终止回合，{a['new_physical_steps']:,} 个物理步；保留前次 60 个回合，合计 120 个 B1 回合、{a['combined_b1_physical_steps']:,} 步。程序失败为 0。独立配对单位仍为 20 个场景。",'',
      'A 为规划状态一致性修复；B 为起点预检、失败重试与缺失分配恢复；C 为已知障碍路径检查与 Hold。V2 是 A+B，不是单独 B。本轮没有修改任何 A/B/C 算法、控制器或环境代码。','',
      '## 实验口径','',
      '- 固定原始索引：0、3、7、10、22、28、31、40、42、46、62、63、67、76、77、78、86、91、95、97。',
      '- 每场环境随机种子为 12729 + 原始索引；400 步上限；任一智能体碰撞即终止。',
      '- 原生 B1 联合分配：`ch3_baseline_bser_prior`；四智能体零残差；不加载模型、不训练。',
      '- 当前、上一轮 B1、本轮运行前后源码身份完全一致；338 个文件、27 条历史来源记录。',
      '- 本轮只新增实验编排、分析、证据和测试，来源清单与历史报告保持不变。',
      '- 149 项既有 SafeSearch 检查与 5 项新增补充实验检查通过；为本地任务范围验证，不是 CI 或全仓测试。','',
      '## B1 六版本','',
      '|版本|开关|Found|完整成功|发现前碰撞|停滞+Hold|有效观测|搜索暴露步数|',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    scores=a['results']['B1']['variants'];delta=a['results']['B1']['comparisons']['V4_minus_V5']
    lines[3:3]=['',
      '**结论：本开发集上，V4=A+B+C 的 Found 最高（8/20），V3=C 的发现前碰撞最少（4/20）。V4相对V5有净改善，但尚未同时达到V3的安全水平。**','',
      f"V4−V5：Found {paired(delta['found'])} 个百分点，发现前碰撞 {paired(delta['pre_found_collision'])} 个百分点，完整成功没有增加。两项差值的95%区间均跨0，不能据此宣称统计上可靠的优势。",'',
      'V1=A 与 V2=A+B 的 Found 均为5/20，发现前碰撞分别13/20、11/20。B在有C和无C时表现不同；本轮支持继续研究组合与安全执行，而非把B视为稳定的单独增益。','']
    for v,s in a['results']['B1']['variants'].items():
        lines.append(f"|{v}|{OPTIONS[v]}|{s['found_count']}/20 ({pct(s['found_rate'])})|{s['success_count']}/20|{s['pre_found_collision_count']}/20 ({pct(s['pre_found_collision_rate'])})|{pct(s['stall_plus_hold_fraction'])}|{pct(s['actual_belief_footprint']['effective_observation_fraction'])}|{s['pre_found_exposure_steps']}|")
    lines+=['','停滞+Hold 的分母是三名搜索者的发现前总 agent-step；有效观测比例的分母为发现前总物理步。有效观测是实际新增或老化网格更新的覆盖代理，不是发现概率。', '',
      '## 配对增量','',
      '下表单位为百分点，括号为 20 场景配对 bootstrap 的 95% percentile 区间（10,000 次，seed=20260928）。区间是描述性的，未做多重比较校正；不是独立正式检验。', '',
      '|比较|解释|Found 差值 [区间]|碰撞差值 [区间]|完整成功差值 [区间]|',
      '|---|---|---:|---:|---:|']
    explanations={'V1_minus_V0':'加入 A','V2_minus_V1':'A 上加入 B','V4_minus_V5':'A+C 上加入 B','V4_minus_V2':'A+B 上加入 C','V4_minus_V0':'全组合与原始','V4_minus_V3':'C 上加入 A+B','V2_minus_V0':'A+B 与原始'}
    for key,d in a['results']['B1']['comparisons'].items():
        lines.append(f"|{key.replace('_minus_',' − ')}|{explanations[key]}|{paired(d['found'])}|{paired(d['pre_found_collision'])}|{paired(d['success'])}|")
    lines+=['','配对区间使用每场景差值；上表汇总停滞与有效观测使用总计数/总暴露，因此不能直接混用两类权重。完整成功的[0,0]区间仅表示本组场景的配对结果完全相同，不能证明总体等效。','',
      '## 发生变化的场景','', '|比较|新增 Found|丢失 Found|新增发现前碰撞|避免发现前碰撞|','|---|---|---|---|---|']
    fmt=lambda x:'、'.join(map(str,x)) or '无'
    for key,d in a['results']['B1']['comparisons'].items():
        lines.append(f"|{key.replace('_minus_',' − ')}|{fmt(d['found']['gained_indices'])}|{fmt(d['found']['lost_indices'])}|{fmt(d['pre_found_collision']['gained_indices'])}|{fmt(d['pre_found_collision']['lost_indices'])}|")
    lines+=['','## B0 与 B1 对照','',
      'B0 只读复用既有 V0–V4 100 回合和 V5 20 回合；本轮没有重跑 B0。既有源码迁移证据链及原始记录已复核。','',
      '|版本|B0 Found|B1 Found|B0 发现前碰撞|B1 发现前碰撞|B0 完整成功|B1 完整成功|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for v in OPTIONS:
        x,y=[a['results'][b]['variants'][v] for b in ('B0','B1')]
        lines.append(f"|{v}|{x['found_count']}/20|{y['found_count']}/20|{x['pre_found_collision_count']}/20|{y['pre_found_collision_count']}/20|{x['success_count']}/20|{y['success_count']}/20|")
    lines+=['','## 诊断计数','',
      '|版本|起点连接失败查询|无效起点查询|Hold 中发生的发现前碰撞|发现后超时数|这些超时的执行者不可达步数 / 发现后步数|',
      '|---|---:|---:|---:|---:|---:|']
    for v,s in a['results']['B1']['variants'].items():
        cases=[r for r in c if r['variant']==v]
        holds=sum(any(h['consecutive_hold_steps_before_collision']>0 for h in r['collision']) for r in cases
                  if r['found_step'] is None or r['found_step'] >= r['physical_steps'])
        timeouts=[r for r in cases if r['found_step'] is not None and r['reason']=='timeout']
        q=s['pre_found_query_reasons']
        lines.append(f"|{v}|{q.get('no_start_connector',0):,}|{q.get('invalid_start',0):,}|{holds}|{len(timeouts)}|{sum(r['post_found']['executor_unreachable_steps'] for r in timeouts)} / {sum(r['post_found']['steps'] for r in timeouts)}|")
    lines+=['','查询次数受重试频率影响，不等于真实不可达场景数。Hold 碰撞并不单独证明水流漂移或制动距离不足；具体机制须结合连续轨迹、速度和位置判断。','']
    case_index={(r['index'],r['variant']):r for r in c}
    pair=read_json(OUT/'pair_mechanisms.json')
    divergence=next(r for r in pair['cases'] if r['index']==3)
    drift=case_index[46,'V4']['collision']
    recovery=next(r for r in gains['cases'] if r['index']==67)['arms']['V4']['observed_recovery_transitions']
    assert len(recovery)==1 and recovery[0]['step']==45
    lines+=['## 具体机制证据','',
      '1. 收益场景：V4相对V5新增场景63、67、77的Found，同时丢失3、95，净增1场。场景67在第45步实际执行了一次成功的SAFE_SEARCH_RECOVERY，第125步Found；V5在第227步发现前碰撞。这证明恢复分支确实参与了成功搜索，但不能把整场收益归结为该单个事件。',
      '2. 场景0：V1与V2都未发现超时，三名搜索者的完成路径停滞区间相同。V2提前拦截无效起点查询后，另外两名搜索者各有可达候选，仍因整次分配要求所有受影响搜索者有路径而拒绝。B减少无效查询，并未在此场景解除健康搜索者的阻塞。',
      f"3. 场景3：V4与V5首次引导状态分歧在第{divergence['first_guidance_difference']['step']}步，首次位置分歧在第{divergence['first_physical_difference']['step']}步。V5当步接受障碍重规划；V4仍用较旧缓存，查询起点连接失败并保持Hold。B改变了刷新和重试时机；该局部事件的记录不能独立证明其单独造成了最后碰撞。"]
    for hit in drift:
        lines.append(f"4. 场景46 V4：智能体{hit['agent']}在连续Hold {hit['consecutive_hold_steps_before_collision']}步、移动{hit['distance_during_final_hold']:.3f}米后，于第{hit['terminal_step']}步碰撞。加入B并未消除该回合的Hold漂移现象。")
    lines+=['','上面场景为结果出现后的探索性机制检查，不是新增的预注册显著性检验。收益场景见 `gain_cases.json`，原始分歧与查询记录见 `pair_mechanisms.json`，终止前连续Hold与路径状态见 `cases.json`。','',
      '## 20 场景逐项结果','',
      '格式：`F发现步 / S成功终止步`、`F发现步 / T400`、`C碰撞步`、`T400`。碰撞角色见 cases.json。','',
      '|原始索引|V0|V1|V2|V3|V4|V5|','|---:|---|---|---|---|---|---|']
    rows={(r['index'],r['variant']):r for r in a['scenario_rows'] if r['baseline']=='B1'}
    for i in sorted({i for i,v in rows}):
        labels=[]
        for v in OPTIONS:
            r=rows[i,v];found=f"F{r['found_step']} / " if r['found_step'] is not None else ''
            term='S' if r['outcome']=='success' else 'C' if 'collision' in r['outcome'] else 'T'
            labels.append(found+term+str(r['physical_steps']))
        lines.append('|'+str(i)+'|'+'|'.join(labels)+'|')
    lines+=['','## 证据与限制','',
      '- `analysis.json`：六版本汇总、配对区间、B1−B0 增量差异、输入哈希。',
      '- `cases.json`：120 个 B1 回合的 Hold、碰撞、完成路径停滞、分配拒绝和发现后执行记录。',
      '- `numeric_audit.json`：独立重算 Found、成功、碰撞及比例分母。',
      '- `../b1_remaining_plan_20260929.json`：运行前冻结的版本、种子、场景及比较设计。',
      '- `../b1_remaining_preparation_20260929.json`：本轮本地检查与运行前源码身份。',
      '- 新原始轨迹：`runs/safe_search_v1/development_B1_remaining_20260929_v1`（Git 忽略）。',
      '- 这 20 个场景已用于开发，不能当作独立测试集。没有运行 HGR、剩余80场景、训练或正式论文验收。',
      '- 所有比较均保留真实终止回合；未将程序错误算成超时，也未剔除物理失败回合。', '']
    target=OUT/'report.md'
    if target.exists():raise ValueError('Do not overwrite a delivered report')
    target.write_text('\n'.join(lines),encoding='utf-8')
    write_json(OUT/'report_binding.json',dict(analysis_sha256=file_hash(OUT/'analysis.json'),cases_sha256=file_hash(OUT/'cases.json'),pair_mechanisms_sha256=file_hash(OUT/'pair_mechanisms.json'),gain_cases_sha256=file_hash(OUT/'gain_cases.json'),writer_sha256=file_hash(__file__),report_sha256=file_hash(target)))
    print(target)


if __name__=='__main__':write()
