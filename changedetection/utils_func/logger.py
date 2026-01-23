import logging
from tabulate import tabulate

class ValidationTableFormatter(logging.Formatter):
    def format(self, record):
        # 假设日志消息是一个字典，包含需要格式化的键值对
        log_info = record.msg if isinstance(record.msg, dict) else {}
        
        # 提取F1_subcls的值并命名
        f1_subcls_values = log_info.get('F1_subcls', [])
        f1_subcls_dict = {
            'no_damage': f1_subcls_values[0] if len(f1_subcls_values) > 0 else None,
            'minor_damage': f1_subcls_values[1] if len(f1_subcls_values) > 1 else None,
            'major_damage': f1_subcls_values[2] if len(f1_subcls_values) > 2 else None,
            'destroy': f1_subcls_values[3] if len(f1_subcls_values) > 3 else None,
        }
        
        # 准备第一行（指标名称）和第二行（对应的值）
        header = (
            f"| {'F1_oa':<12} | {'F1_loc':<12} | {'F1_bda':<12} | "
            f"{'no_damage':<12} | {'minor_damage':<12} | {'major_damage':<12} | {'destroy':<12} |"
        )
        values = (
            f"| {log_info.get('F1_oa', 'N/A'):<12} | {log_info.get('F1_loc', 'N/A'):<12} | {log_info.get('F1_bda', 'N/A'):<12} | "
            f"{f1_subcls_dict['no_damage']:<12} | {f1_subcls_dict['minor_damage']:<12} | {f1_subcls_dict['major_damage']:<12} | {f1_subcls_dict['destroy']:<12} |"
        )
        
        return f"{header}\n{values}"
    
def log_validation_metrics(logger, metrics, final=False):
    f1_subcls_values = metrics.get('F1_subcls', [])
    # f1_subcls_dict = {
    #     'no_damage': round(f1_subcls_values[0], 4) if len(f1_subcls_values) > 0 else None,
    #     'minor_damage': round(f1_subcls_values[1], 4) if len(f1_subcls_values) > 1 else None,
    #     'major_damage': round(f1_subcls_values[2], 4) if len(f1_subcls_values) > 2 else None,
    #     'destroy': round(f1_subcls_values[3], 4) if len(f1_subcls_values) > 3 else None,
    # }
    
    table_data = [
        ["Metric", "F1_oa", "F1_loc", "F1_bda", "no_damage", "minor_damage", "major_damage", "destroy"],
        ["Value", metrics['F1_oa'], metrics['F1_loc'], metrics['F1_bda'], f1_subcls_values[0], 
        f1_subcls_values[1], f1_subcls_values[2], f1_subcls_values[3]]]
    # table = tabulate(table_data, headers='firstrow', tablefmt="grid", floatfmt=".4f", stralign="center", numalign="center")
    table = tabulate(table_data, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        # logger.info("The best accuracy:\n" + table)
        logger.info(f"The best accuracy:\n{table}")
    else:
        logger.info("\n" + table)
    # header = (
    #     f"| {'F1_oa':^12} | {'F1_loc':^12} | {'F1_bda':^12} | "
    #     f"{'no_damage':^12} | {'minor_damage':^12} | {'major_damage':^12} | {'destroy':^12} |"
    #     )
    # values = (
    #     f"| {round(metrics['F1_oa'], 4):^12} | {round(metrics['F1_loc'], 4):^12} | {round(metrics['F1_bda'], 4):^12} | "
    #     f"{f1_subcls_dict['no_damage']:^12} | {f1_subcls_dict['minor_damage']:^12} | {f1_subcls_dict['major_damage']:^12} | {f1_subcls_dict['destroy']:^12} |"
    #     )

    # logger.info("\n" + header + "\n" + values)

def log_dfc_metrics(logger, metrics, final=False, test=False):
    f1_subcls_values = metrics.get('F1_subcls', [])
    table_data = [
        ["Metric", "F1_oa", "F1_loc", "F1_bda", "intact", "damaged", "destroy"],
        ["Value", metrics['F1_oa'], metrics['F1_loc'], metrics['F1_bda'], f1_subcls_values[0], 
        f1_subcls_values[1], f1_subcls_values[2]]]
    table = tabulate(table_data, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        logger.info(f"The best accuracy:\n{table}")
    else:
        logger.info("\n" + table)

def log_metrics(logger, metrics, final=False):
    iou_subcls = metrics.get('iou_subcls', [])
    table_data = [
        ["Metric", "F1_loc", "F1_bda", "Final_OA", "mIoU", "background", "intact", "damaged", "destroy"],
        ["Value", metrics['F1_loc'], metrics['F1_bda'], metrics['oa'], metrics["mIoU"],
         iou_subcls[0],  iou_subcls[1], iou_subcls[2], iou_subcls[3]]]
    table = tabulate(table_data, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        logger.info(f"The best accuracy:\n{table}")
    else:
        logger.info("\n" + table)

def log_corn_metrics(logger, metrics, final=False, test=False):
    f1_subcls_values = metrics.get('F1_subcls', [])
    if test:
        table_data = [
            ["Metric", "Acc", "F1_oa", "loc_acc", "dmg_acc", "no_damage", "damage"],
            ["Value", metrics['acc'], metrics['F1_oa'], metrics['loc_acc'], metrics['dmg_acc'],
             f1_subcls_values[0], f1_subcls_values[1]]]
    else:
        table_data = [
            ["Metric", "F1_oa", "F1_loc", "F1_bda", "no_damage", "damage"],
            ["Value", metrics['F1_oa'], metrics['F1_loc'], metrics['F1_bda'], f1_subcls_values[0], 
            f1_subcls_values[1]]]
    table = tabulate(table_data, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        logger.info(f"The best accuracy:\n{table}")
    else:
        logger.info("\n" + table)

def log_bcd_metrics(logger, metrics, final=False, test=False):
    table_date=[
        ["Metric", "mAcc", "F1-score", "mIoU", 'Precision', 'Recall'],
        ["Value", metrics['macc'], metrics['F1_oa'], metrics['mIoU'], metrics['precision'], metrics['recall']]
        ]
    table = tabulate(table_date, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        logger.info(f"The best accuracy:\n{table}")
    elif test:
        logger.info(f"The test performance:\n{table}")
    else:
        logger.info(f"\n{table}")
def log_scd_metrics(logger, metrics, final=False, test=False):
    table_date=[
        ["Metric", "kappa", "F1-score", "mIoU", 'Sek', 'acc'],
        ["Value", metrics['kappa'], metrics['F1_oa'], metrics['mIoU'], metrics['Sek'], metrics['acc']]
        ]
    table = tabulate(table_date, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    if final:
        logger.info(f"The best accuracy:\n{table}")
    elif test:
        logger.info(f"The test performance:\n{table}")
    else:
        logger.info(f"\n{table}")
    
def log_test_metrics(logger, metrics):
    f1_subcls_values = metrics.get('F1_subcls', [])
    table_data = [
        ["Metric", "Acc", "Uoc_index", "F1_oa", "F1_loc", "F1_bda", "no_damage", "minor_damage", "major_damage", "destroy"],
        ["Value", metrics['acc'], metrics['uoc'], metrics['F1_oa'], metrics['F1_loc'], metrics['F1_bda'], f1_subcls_values[0], 
        f1_subcls_values[1], f1_subcls_values[2], f1_subcls_values[3]]]
    table = tabulate(table_data, headers='firstrow', tablefmt="grid", stralign="center", numalign="center")
    logger.info(f"The test performance:\n{table}")


def get_logger(log_file='train.log', log_level=logging.DEBUG, test=False):
    """
    设置日志配置，将日志输出到文件和控制台。

    Args:
        log_file (str): 日志文件路径
        log_level (int): 日志级别
    """
    filemode = 'a' if test else 'w'
    # 配置基本的日志设置
    logging.basicConfig(filename=log_file, 
                        filemode=filemode, 
                        level=log_level, 
                        format='%(asctime)s - %(levelname)s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # 获取默认的logger
    logger = logging.getLogger()

    # 创建一个StreamHandler来输出日志消息到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 设置控制台输出的格式
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # 将控制台Handler添加到Logger
    logger.addHandler(console_handler)
    
    return logger