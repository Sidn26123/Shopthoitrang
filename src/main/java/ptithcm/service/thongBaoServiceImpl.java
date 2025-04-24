package ptithcm.service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.entity.ThongBaoEntity;

import ptithcm.dao.thongBaoDAO;


@Transactional
@Service
public class thongBaoServiceImpl implements thongBaoService {
	@Autowired
	thongBaoDAO tbDAO;
	
	@Override
	public List<ThongBaoEntity> LayThongBaoCuaUser(int mand){
		return tbDAO.layThongBaoCuaUser(mand);
	}
	
	@Override
	public Integer LayTongThongBaoChuaDoc(List<ThongBaoEntity> listTB) {
		Integer count = 0;
		
		for (ThongBaoEntity tb : listTB) {
			if (!tb.isRead()) {
				count = count++;
			} 
		}
		return count;
	}
	
	@Override
	public void addThongBao(ThongBaoEntity thongBao) {
		tbDAO.addThongBao(thongBao);
	}
	
	@Override
	public void deleteThongBao(ThongBaoEntity thongbao) {
		tbDAO.deleteThongBao(thongbao);
	}
	
	@Override
	public void updateThongBao(ThongBaoEntity thongBao) {
		tbDAO.updateThongBao(thongBao);
	}
	
	@Override
	public void markAllNotificationRead(int mand) {
		tbDAO.markAllNotificationRead(mand);
	}

	@Override
	public int getNumNotificationUnread(List<ThongBaoEntity> listTB) {
		int count = 0;
		for (ThongBaoEntity tb : listTB) {
			if (!tb.isRead()) {
				count++;
			}
		}
		return count;
	}
}