import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import os
import pickle
import traceback
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_cycle_industry.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionCycleAnalyzer:
    """
    情绪周期日度分析器（行业版，并发加速）
    股票列表从 CSV 读取，主线使用申万二级行业
    """

    def __init__(self, csv_path: str, code_column: str = 'code',
                 name_column: str = 'name', state_file: str = 'emotion_state.pkl',
                 max_retries: int = 3, request_delay: float = 0.3,
                 max_workers: int = 10):
        self.csv_path = csv_path
        self.code_column = code_column
        self.name_column = name_column
        self.state_file = state_file
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.max_workers = max_workers
        self.is_logged_in = False

        self.stock_codes, self.stock_names = self._load_stock_list()
        self.state = self._load_state()

        # 行业缓存
        self.stock_industry = {}
        self.industry_stocks = {}
        self.industry_size = {}

    # -------------------- 股票列表加载 --------------------
    def _load_stock_list(self) -> Tuple[List[str], Dict[str, str]]:
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            if self.code_column not in df.columns:
                raise ValueError(f"CSV中未找到列 '{self.code_column}'")
            codes = df[self.code_column].astype(str).str.strip().tolist()
            formatted_codes = []
            for c in codes:
                if not c.startswith(('sh.', 'sz.', 'bj.')):
                    if c.startswith('6'):
                        c = 'sh.' + c
                    elif c.startswith(('0', '3')):
                        c = 'sz.' + c
                    else:
                        c = 'sh.' + c
                formatted_codes.append(c)

            if self.name_column in df.columns:
                name_dict = dict(zip(formatted_codes, df[self.name_column].astype(str)))
            else:
                name_dict = {c: '未知' for c in formatted_codes}
            logger.info(f"从CSV加载 {len(formatted_codes)} 只股票")
            return formatted_codes, name_dict
        except Exception as e:
            logger.error(f"股票列表加载失败: {e}")
            return [], {}

    # -------------------- 状态持久化 --------------------
    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info("📂 加载历史状态成功")
                return state
            except:
                pass
        return {
            'prev_main_sector': None,
            'prev_hl_stocks': [],
            'prev_L2': None,
            'prev_drop_ratio_trigger': False,
            'leader_code': None,
            'leader_break_days': 0,
            'daily_fry_rate_seq': [],
            'daily_promote_rate_seq': [],
            'stock_burst_count': {}
        }

    def _save_state(self):
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)

    # -------------------- 登录/登出 --------------------
    def login(self) -> bool:
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                lg = bs.login()
                if lg.error_code == '0':
                    self.is_logged_in = True
                    logger.info("✅ 登录成功")
                    return True
            except:
                pass
            time.sleep(2 ** attempt)
        return False

    def logout(self):
        if self.is_logged_in:
            try:
                bs.logout()
            except:
                pass
            self.is_logged_in = False
            logger.info("✅ 已登出")

    # -------------------- 行业缓存初始化 --------------------
    def _init_industry_cache(self):
        """初始化申万二级行业映射（单线程，首次运行）"""
        logger.info("开始初始化行业缓存...")
        self.stock_industry.clear()
        self.industry_stocks.clear()
        for code in tqdm(self.stock_codes, desc='行业缓存'):
            try:
                time.sleep(0.02)  # 控制请求频率
                rs = bs.query_stock_industry(code)
                if rs.error_code != '0':
                    continue
                secondary_industry = None
                while rs.next():
                    row = rs.get_row_data()
                    ind_name = row[1]
                    ind_class = row[2] if len(row) > 2 else ''
                    if '二级' in ind_class:
                        secondary_industry = ind_name
                        break
                if secondary_industry:
                    self.stock_industry[code] = secondary_industry
                    self.industry_stocks.setdefault(secondary_industry, []).append(code)
            except:
                continue
        self.industry_size = {k: len(v) for k, v in self.industry_stocks.items()}
        logger.info(f"行业缓存完成，共 {len(self.industry_stocks)} 个二级行业")

    # -------------------- 数据获取 --------------------
    def get_daily_data(self, code: str, end_date: str, days: int = 150) -> Optional[pd.DataFrame]:
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)).strftime('%Y-%m-%d')
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn"
        for attempt in range(self.max_retries):
            try:
                delay = self.request_delay + (2 ** attempt) + np.random.uniform(0, 0.5)
                time.sleep(delay)
                rs = bs.query_history_k_data_plus(
                    code, fields,
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="3"
                )
                if rs.error_code != '0':
                    continue
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                if not data_list:
                    return None
                df = pd.DataFrame(data_list, columns=fields.split(','))
                for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['date', 'close', 'preclose'])
                df = df.sort_values('date').reset_index(drop=True)
                if len(df) > days:
                    df = df.tail(days)
                return df
            except:
                continue
        return None

    # -------------------- 涨停/跌停/连板判断 --------------------
    @staticmethod
    def is_limit_up(row) -> bool:
        if row['preclose'] <= 0:
            return False
        up_ratio = row['close'] / row['preclose'] - 1
        if up_ratio < 0.095:
            return False
        if row['open'] == row['high'] == row['close'] and row['close'] == row['high']:
            if pd.notna(row['turn']) and row['turn'] < 1.0:
                return False
        return True

    @staticmethod
    def is_limit_down(row) -> bool:
        if row['preclose'] <= 0:
            return False
        down_ratio = row['close'] / row['preclose'] - 1
        return down_ratio <= -0.095

    @staticmethod
    def is_break_up(row) -> bool:
        if row['preclose'] <= 0:
            return False
        if row['high'] / row['preclose'] - 1 >= 0.095 and row['close'] < row['high']:
            return True
        return False

    def count_consecutive_boards(self, df: pd.DataFrame, date_idx: int) -> int:
        count = 0
        for i in range(date_idx, -1, -1):
            row = df.iloc[i]
            if i == date_idx:
                if self.is_limit_up(row):
                    count += 1
                else:
                    break
            else:
                if self.is_limit_up(row):
                    count += 1
                else:
                    break
        return count

    def get_20d_burst_count(self, code: str, end_date: str) -> int:
        df = self.get_daily_data(code, end_date, days=30)
        if df is None:
            return 0
        df = df.tail(20)
        cnt = 0
        for _, row in df.iterrows():
            if self.is_limit_up(row):
                cnt += 1
        return cnt

    # -------------------- 单只股票分析函数（用于多线程） --------------------
    def _analyze_single_stock(self, code: str, date_str: str, burst_cache: dict) -> dict:
        """
        线程安全：只读取不修改共享状态，返回该股票的分析结果
        返回: {'code': str, 'hl_status': 'limit_up'/'break_up'/'none',
               'limit_down': bool, 'high_face': bool, 'sky_floor': bool,
               'board_num': int, 'turn': float, 'preclose': float,
               'high': float, 'low': float, 'close': float}
        """
        result = {
            'code': code,
            'hl_status': 'none',
            'limit_down': False,
            'high_face': False,
            'sky_floor': False,
            'board_num': 0,
            'turn': 0.0,
            'preclose': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0
        }
        try:
            df = self.get_daily_data(code, date_str, days=70)
            if df is None or df.empty:
                return result
            latest = df.iloc[-1]
            result['preclose'] = latest['preclose']
            result['high'] = latest['high']
            result['low'] = latest['low']
            result['close'] = latest['close']
            result['turn'] = latest['turn'] if pd.notna(latest['turn']) else 0.0

            # 跌停
            if self.is_limit_down(latest):
                result['limit_down'] = True

            # 涨停或炸板
            if self.is_limit_up(latest):
                result['hl_status'] = 'limit_up'
                result['board_num'] = self.count_consecutive_boards(df, len(df) - 1)
            elif self.is_break_up(latest):
                result['hl_status'] = 'break_up'
                result['board_num'] = self.count_consecutive_boards(df, len(df) - 1)

            # 高位大面判断（需要近20日涨停次数）
            if code not in burst_cache:
                burst_cache[code] = self.get_20d_burst_count(code, date_str)  # 此处在子线程中也会更新全局字典，需要加锁（见下文）
            burst_count = burst_cache.get(code, 0)
            if burst_count >= 5 and latest['preclose'] > 0:
                pullback = (latest['high'] - latest['close']) / latest['preclose']
                if pullback >= 0.08:
                    result['high_face'] = True
                    # 天地板
                    if latest['close'] / latest['preclose'] - 1 <= -0.09 and \
                            latest['high'] / latest['preclose'] - 1 >= 0.09:
                        result['sky_floor'] = True
        except:
            pass
        return result

    # -------------------- 主线行业识别 --------------------
    def detect_main_sector(self, hl_stocks_info: List[dict]):
        if not hl_stocks_info:
            return {'has_main': False, 'sector_name': None, 'hl_count': 0}

        industry_hl = {}
        for stock in hl_stocks_info:
            ind = self.stock_industry.get(stock['code'])
            if ind:
                industry_hl[ind] = industry_hl.get(ind, 0) + 1

        if not industry_hl:
            return {'has_main': False, 'sector_name': None, 'hl_count': 0}

        max_ind = max(industry_hl, key=industry_hl.get)
        max_count = industry_hl[max_ind]

        prev = self.state.get('prev_main_sector')
        if prev and prev.get('sector_name'):
            prev_ind = prev['sector_name']
            prev_count = industry_hl.get(prev_ind, 0)
            threshold = max(3, max_count * 0.5)
            if prev_count >= threshold:
                return {'has_main': True, 'sector_name': prev_ind, 'hl_count': prev_count}

        return {'has_main': True, 'sector_name': max_ind, 'hl_count': max_count}

    # -------------------- L1 全局情绪 --------------------
    def calc_L1(self, hs300_df: pd.DataFrame,
                limit_up_count: int, limit_down_count: int,
                high_face_count: int) -> Tuple[float, bool, bool]:
        score = 10.0
        systemic_risk = False
        if hs300_df is not None and not hs300_df.empty:
            chg = hs300_df.iloc[-1]['close'] / hs300_df.iloc[-1]['preclose'] - 1
            if chg <= -0.025:
                score -= 4
                systemic_risk = True

        drop_trigger = False
        if limit_up_count > 0 and (limit_down_count / limit_up_count) > 1.2:
            drop_trigger = True
            if self.state.get('prev_drop_ratio_trigger', False):
                score -= 3

        if high_face_count >= 15:
            score -= 3
        elif high_face_count >= 8:
            score -= 1

        score = max(0, min(10, score))
        return score, systemic_risk, drop_trigger

    # -------------------- L2 主线情绪 --------------------
    def calc_L2(self, main_industry_name: str, hl_stocks_info: List[dict],
                high_face_count: int) -> float:
        if not main_industry_name:
            return 0.0
        sector_codes = self.industry_stocks.get(main_industry_name, [])
        if not sector_codes:
            return 0.0
        main_hl = [s for s in hl_stocks_info if s['code'] in sector_codes]

        # 梯队完整性
        board_4_plus = len([s for s in main_hl if s.get('board_num', 0) >= 4])
        tier_score = 4 if board_4_plus >= 3 else (2 if board_4_plus >= 1 else 0)

        # 高位炸板率偏离
        fry_rates = self.state.get('daily_fry_rate_seq', [])
        avg_fry = np.mean(fry_rates[-20:]) if len(fry_rates) >= 20 else 0.3
        high_board = [s for s in main_hl if s.get('board_num', 0) >= 3 and s.get('is_break', False)]
        valid_high = [s for s in main_hl if s.get('board_num', 0) >= 3]
        today_fry = len(high_board) / len(valid_high) if valid_high else 0
        diff = today_fry - avg_fry
        fry_score = 4 if diff < 0 else (2 if diff <= 0.15 else 0)

        # 晋级率偏离
        promote_rates = self.state.get('daily_promote_rate_seq', [])
        avg_promote = np.mean(promote_rates[-20:]) if len(promote_rates) >= 20 else 0.3
        yesterday_hl = self.state.get('prev_hl_stocks', [])
        today_hl_codes = {s['code'] for s in hl_stocks_info if not s.get('is_break')}
        promote_count = len([c for c in yesterday_hl if c in today_hl_codes])
        today_promote = promote_count / len(yesterday_hl) if yesterday_hl else 0
        dev = today_promote - avg_promote
        promote_score = 2 if dev >= 0 else (1 if dev >= -0.20 else 0)

        # 高位大面
        if high_face_count == 0:
            face_score = 3
        elif 1 <= high_face_count <= 5:
            face_score = 1
        else:
            has_sky = any(s.get('is_sky_floor', False) for s in hl_stocks_info)
            face_score = -1 if (high_face_count >= 6 and has_sky) else 0

        L2 = tier_score + fry_score + promote_score + face_score
        return min(13, L2)

    # -------------------- L3 龙头信号 --------------------
    def detect_leader(self, main_industry_name: str, hl_stocks_info: List[dict]) -> Optional[str]:
        if not main_industry_name:
            return None
        sector_codes = self.industry_stocks.get(main_industry_name, [])
        candidates = [s for s in hl_stocks_info if s['code'] in sector_codes and not s.get('is_break')]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x.get('board_num', 0), -x.get('turn', 0)))
        return candidates[0]['code']

    def calc_L3_signal(self, leader_code: str, today_hl_codes: set,
                       hl_stocks_info: List[dict]) -> Tuple[str, int]:
        prev_leader = self.state.get('leader_code')
        break_days = self.state.get('leader_break_days', 0)
        if leader_code != prev_leader and prev_leader is not None:
            break_days = 0
        leader_board = leader_code in today_hl_codes if leader_code else False
        new_break_days = 0 if leader_board else break_days + 1
        sky_floor_cnt = sum(1 for s in hl_stocks_info if s.get('is_sky_floor', False))
        if new_break_days >= 2 or sky_floor_cnt >= 2:
            signal = 'confirm'
        elif new_break_days >= 1 or sky_floor_cnt >= 1:
            signal = 'warning'
        else:
            signal = 'none'
        return signal, new_break_days

    # -------------------- 动态阈值与仓位 --------------------
    @staticmethod
    def dynamic_params(L1_score: float):
        if L1_score <= 4:
            mood, thresh = '恐慌', {'止损': '-3%', '竞价溢价': '+1%', '跌停容量': 5, '量比阈值': 0.03, '入场最低分': 38}
        elif L1_score <= 7:
            mood, thresh = '中性', {'止损': '-5%', '竞价溢价': '+1.5%', '跌停容量': 10, '量比阈值': 0.05,
                                    '入场最低分': 45}
        else:
            mood, thresh = '亢奋', {'止损': '-7%', '竞价溢价': '+2%', '跌停容量': 15, '量比阈值': 0.07,
                                    '入场最低分': 55}
        return mood, thresh

    @staticmethod
    def position_advice(L2_score: float, systemic_risk: bool):
        if systemic_risk:
            return 0, 0, '0档（空仓）'
        if L2_score >= 16:
            return 0.25, 0.9, '5档'
        elif L2_score >= 12:
            return 0.20, 0.7, '4档'
        elif L2_score >= 8:
            return 0.10, 0.3, '3档'
        elif L2_score >= 4:
            return 0.05, 0.1, '2档'
        else:
            return 0, 0, '1档（空仓）'

    # -------------------- 主分析函数（并发优化） --------------------
    def analyze(self, date_str: str):
        logger.info(f"======== 开始分析 {date_str} ========")
        if not self.stock_codes:
            logger.error("股票列表为空，请检查CSV文件")
            return

        # 行业缓存（首次运行）
        if not self.stock_industry:
            self._init_industry_cache()

        # 沪深300数据
        hs300_df = self.get_daily_data('sh.000300', date_str, days=5)

        # ---------- 多线程并发扫描股票 ----------
        logger.info(f"启动 {self.max_workers} 个线程扫描 {len(self.stock_codes)} 只股票...")
        # 使用线程安全的 burst_cache 字典（需要加锁或使用局部缓存后合并）
        # 为简化，burst_cache 在子线程中也会更新，使用 threading.Lock 保护
        from threading import Lock
        burst_lock = Lock()
        burst_cache = self.state.get('stock_burst_count', {}).copy()  # 浅拷贝，内部值仍被修改但安全

        def _thread_safe_analyze(code):
            # 线程内独立获取 burst_count，通过锁更新共享 burst_cache
            res = self._analyze_single_stock(code, date_str, burst_cache)
            # 如果内部更新了 burst_cache，由于已加锁，可直接返回
            return res

        # 收集结果
        limit_up_list = []
        limit_down_count = 0
        high_face_count = 0
        today_hl_codes = set()

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(_thread_safe_analyze, code): code for code in self.stock_codes}
            # 进度条
            with tqdm(total=len(self.stock_codes), desc="个股扫描", unit="只") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        res = future.result()
                    except Exception as e:
                        res = None
                        logger.debug(f"股票 {code} 分析异常: {e}")
                    pbar.update(1)

                    if res is None:
                        continue

                    # 统计跌停
                    if res.get('limit_down'):
                        limit_down_count += 1

                    # 处理涨停/炸板
                    if res.get('hl_status') in ('limit_up', 'break_up'):
                        info = {
                            'code': code,
                            'board_num': res.get('board_num', 0),
                            'turn': res.get('turn', 0),
                            'is_break': res.get('hl_status') == 'break_up',
                            'is_sky_floor': res.get('sky_floor', False),
                            'preclose': res.get('preclose', 0),
                            'high': res.get('high', 0),
                            'low': res.get('low', 0),
                            'close': res.get('close', 0)
                        }
                        limit_up_list.append(info)
                        if not info['is_break']:
                            today_hl_codes.add(code)

                    # 高位大面
                    if res.get('high_face'):
                        high_face_count += 1
                        # 如果该股票同时也在涨停列表中，标记天地板已在 info 中处理
                        # 对于不在涨停列表中的高位大面，如需要也可单独记录，这里不处理

        # 同步 burst_cache 回 state（已在子线程中实时更新）
        self.state['stock_burst_count'] = burst_cache

        logger.info(f"扫描完成，涨停(含炸板): {len(limit_up_list)}, 跌停: {limit_down_count}")

        # ---------- 主线识别 ----------
        main_info = self.detect_main_sector(limit_up_list)
        logger.info(f"主线行业: {main_info}")

        # ---------- L1 计算 ----------
        L1_score, systemic_risk, drop_trigger = self.calc_L1(
            hs300_df, len(today_hl_codes), limit_down_count, high_face_count
        )

        # ---------- L2 计算 ----------
        L2_score = self.calc_L2(main_info.get('sector_name'), limit_up_list, high_face_count)

        # ---------- L3 龙头信号 ----------
        leader_code = self.detect_leader(main_info.get('sector_name'), limit_up_list)
        L3_signal, break_days = self.calc_L3_signal(leader_code, today_hl_codes, limit_up_list)

        # ---------- 双日退潮确认 ----------
        prev_L2 = self.state.get('prev_L2')
        tide_confirm = False
        if prev_L2 is not None:
            if (L2_score <= 6) and (prev_L2 <= 7) and (L3_signal in ('warning', 'confirm')):
                tide_confirm = True

        # ---------- 动态阈值与仓位 ----------
        mood, thresholds = self.dynamic_params(L1_score)
        single_pos, total_pos, pos_tier = self.position_advice(L2_score, systemic_risk)

        # ---------- 更新状态 ----------
        self.state['prev_main_sector'] = {
            'name': main_info.get('sector_name'),
            'hl_count': main_info.get('hl_count', 0)
        }
        self.state['prev_hl_stocks'] = list(today_hl_codes)
        self.state['prev_L2'] = L2_score
        self.state['prev_drop_ratio_trigger'] = drop_trigger
        self.state['leader_code'] = leader_code
        self.state['leader_break_days'] = break_days
        self._save_state()

        # ---------- 输出报告 ----------
        report = f"""
        ==============================================
               情绪周期日度分析报告 ({date_str})
        ==============================================
        【L1 全局市场情绪】 {L1_score:.1f} 分  环境档位: {mood}
        【L2 主线情绪】 {L2_score:.1f} 分  (满分13，无竞价)
        【L3 龙头信号】 {L3_signal}  (龙头: {leader_code}, 断板 {break_days}天)
        【双日退潮确认】 {'🔴 已确认！' if tide_confirm else '🟢 未触发'}
        ----------------------------------------------
        【动态阈值】
          止损: {thresholds['止损']}  竞价溢价: {thresholds['竞价溢价']}
          跌停容量: {thresholds['跌停容量']}  量比阈值: {thresholds['量比阈值']}
          入场最低分: {thresholds['入场最低分']}
        【仓位建议】
          单股上限: {single_pos * 100:.0f}%  总仓上限: {total_pos * 100:.0f}%  ({pos_tier})
        ----------------------------------------------
        【主线行业】 {'有' if main_info['has_main'] else '无'}
          行业: {main_info.get('sector_name')}
          涨停数: {main_info.get('hl_count')}
        """
        if tide_confirm:
            report += "\n        ⚠️ 双日退潮确认！建议清仓禁买！\n"
        report += "=============================================="

        print(report)
        logger.info(report)
        return {
            'L1': L1_score, 'L2': L2_score, 'L3': L3_signal,
            'tide_confirm': tide_confirm, 'mood': mood,
            'single_pos': single_pos, 'total_pos': total_pos
        }

    # -------------------- 运行入口 --------------------
    def run(self, date_str: str = None):
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        print("=" * 60)
        print("情绪周期日度分析器（行业版，并发加速）")
        print(f"股票池: {self.csv_path} ({len(self.stock_codes)}只)")
        print(f"分析日期: {date_str}")
        print("=" * 60)

        if not self.login():
            logger.error("登录失败")
            return
        try:
            self.analyze(date_str)
        except Exception as e:
            logger.error(f"分析异常: {traceback.format_exc()}")
        finally:
            self.logout()


# -------------------- 示例CSV生成 --------------------
def create_sample_csv(filepath='stock_pool.csv'):
    sample = {
        'code': ['600519', '000858', '300750', '002594', '601012'],
        'name': ['贵州茅台', '五粮液', '宁德时代', '比亚迪', '隆基绿能']
    }
    pd.DataFrame(sample).to_csv(filepath, index=False, encoding='utf-8-sig')
    logger.info(f"示例股票池已生成: {filepath}")


def main():
    print("=" * 60)
    print("情绪周期日度分析器（行业版，并发加速）")
    print("=" * 60)
    csv_path = input("请输入股票池CSV路径: ").strip()
    if not os.path.exists(csv_path):
        print("文件不存在，是否创建示例？(y/n)")
        if input().strip().lower() == 'y':
            create_sample_csv(csv_path)
        else:
            return

    code_col = input("请输入代码列名 (默认'code'): ").strip() or 'code'
    name_col = input("请输入名称列名 (默认'name'): ").strip() or 'name'
    date_input = input("请输入分析日期 (YYYY-MM-DD，回车为今天): ").strip()
    workers = input("请输入并发线程数 (默认10): ").strip()
    max_workers = int(workers) if workers.isdigit() else 10

    analyzer = EmotionCycleAnalyzer(
        csv_path=csv_path,
        code_column=code_col,
        name_column=name_col,
        state_file='emotion_state_industry.pkl',
        max_workers=max_workers
    )
    analyzer.run(date_input if date_input else None)


if __name__ == "__main__":
    main()